# Copyright (c) 2021 NVIDIA CORPORATION
#               2023 58.com(Wuba) Inc AI Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import triton_python_backend_utils as pb_utils
import numpy as np
import multiprocessing
from swig_decoders import ctc_beam_search_decoder_batch, \
    Scorer, HotWordsScorer, PathTrie, TrieVector, map_batch
import json
import os
import yaml
import jieba
from lib import utils
import re
from torch.utils.dlpack import from_dlpack


class usePunctuator:

    def __init__(self, use_cuda=0, language="en"):
        self.lan = language
        self.use_cuda = use_cuda
        self.word2id = {}
        self.punc2id = {}
        self.class2punc = {}
        self.model = None

    def load_data(self, path):
        try:
            in_vocab_path = os.path.join(path, "vocab")
            self.word2id = utils.load_vocab(in_vocab_path,
                                            extra_word_list=["<UNK>", "<END>"])

            out_vocab_path = os.path.join(path, "punc_vocab")
            self.punc2id = utils.load_vocab(out_vocab_path,
                                            extra_word_list=[" "])
            # print("punc2id",self.punc2id)
            self.class2punc = {k: v for (v, k) in self.punc2id.items()}
        except Exception as e:
            print(e)
            print("********** load error *********")

    def is_number(self, x):
        """替换数字"""
        pattern_numbers = re.compile(r"\d")
        return len(pattern_numbers.sub("", x)) / len(x) < 0.6

    def is_emoji(self, x):
        """替换表情"""
        pattern_emojis = re.compile(r"^[#][\d]+[#]?[\d]*$")
        return pattern_emojis.findall(x)

    def _preprocess(self, txt_seq):
        """Convert txt sequence to word-id-seq."""
        input = []
        if isinstance(txt_seq, str):
            txt_seq = txt_seq.split()
#        print(txt_seq)
        for token in txt_seq:
            if self.is_emoji(token):
                # num_list.append(token)
                token = "<emoji>"
            elif self.is_number(token):
                # num_list.append(token)
                token = "<NUM>"
#                print(token, self.word2id.get(token, self.word2id["<UNK>"]), self.word2id)
            input.append(self.word2id.get(token, self.word2id["<UNK>"]))
        input.append(self.word2id["<END>"])
        # input = torch.LongTensor(input)
        return input

    def handle(self, text):
        if not text:
            return text
        word_id_seq = self._preprocess(text)
        input_lengths = np.array([[len(word_id_seq)]]).astype(np.int32)
        input = np.array([word_id_seq]).astype(np.int32)
        # print('input.shape: ', input.shape)
        # print('input_lengths.shape: ', input_lengths.shape)

        in_tensor_0 = pb_utils.Tensor('input', input)
        in_tensor_1 = pb_utils.Tensor('input_lengths', input_lengths)
        input_tensors = [in_tensor_0, in_tensor_1]
        inference_request = pb_utils.InferenceRequest(
            model_name='biaodian',
            requested_output_names=['score'],
            inputs=input_tensors)

        inference_response = inference_request.exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(
                inference_response.error().message())
        else:
            # Extract the output tensors from the inference response.
            score = pb_utils.get_output_tensor_by_name(
                inference_response, 'score')
            score = from_dlpack(score.to_dlpack())
            predict = np.argmax(score.cpu().numpy(), 2)
            predict = predict.tolist()
        return predict[0]


class TritonPythonModel:

    def initialize(self, args):
        self.model_config = model_config = json.loads(args['model_config'])
        self.max_batch_size = max(model_config["max_batch_size"], 1)

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT0")
        # Convert Triton types to numpy types
        self.out0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])

        # Get INPUT configuration
        batch_log_probs = pb_utils.get_input_config_by_name(
            model_config, "batch_log_probs")
        self.beam_size = batch_log_probs['dims'][-1]
        self.data_type = np.float32
        self.lm = None
        self.hotwords_scorer = None
        self.init_ctc_rescore(self.model_config['parameters'])

        jieba.initialize()
        jieba.load_userdict('biaodian/zh_punctuation/jieba_dict.txt')
        self.use_punctuator = usePunctuator(use_cuda=0, language='zh')
        self.use_punctuator.load_data('biaodian/zh_punctuation')

        print('Initialized CTC!')

    def init_ctc_rescore(self, parameters):
        num_processes = multiprocessing.cpu_count()
        cutoff_prob = 0.9999
        blank_id = 0
        alpha = 2.0
        beta = 1.0
        bidecoder = 0
        lm_path, vocab_path = None, None
        for li in parameters.items():
            key, value = li
            value = value["string_value"]
            if key == "num_processes":
                num_processes = int(value)
            elif key == "blank_id":
                blank_id = int(value)
            elif key == "cutoff_prob":
                cutoff_prob = float(value)
            elif key == "lm_path":
                lm_path = value
            elif key == "hotwords_path":
                hotwords_path = value
            elif key == "alpha":
                alpha = float(value)
            elif key == "beta":
                beta = float(value)
            elif key == "vocabulary":
                vocab_path = value
            elif key == "bidecoder":
                bidecoder = int(value)

        self.num_processes = num_processes
        self.cutoff_prob = cutoff_prob
        self.blank_id = blank_id
        _, vocab = self.load_vocab(vocab_path)
        if lm_path and os.path.exists(lm_path):
            self.lm = Scorer(alpha, beta, lm_path, vocab)
            print("Successfully load language model!")
        if hotwords_path and os.path.exists(hotwords_path):
            self.hotwords = self.load_hotwords(hotwords_path)
            max_order = 4
            if self.hotwords is not None:
                for w in self.hotwords:
                    max_order = max(max_order, len(w))
                self.hotwords_scorer = HotWordsScorer(self.hotwords,
                                                      vocab,
                                                      window_length=max_order,
                                                      SPACE_ID=-2,
                                                      is_character_based=True)
                print(
                    f"Successfully load hotwords! Hotwords orders = {max_order}"
                )
        self.vocabulary = vocab
        self.bidecoder = bidecoder
        sos = eos = len(vocab) - 1
        self.sos = sos
        self.eos = eos

    def load_vocab(self, vocab_file):
        """
        load lang_char.txt
        """
        id2vocab = {}
        with open(vocab_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                char, id = line.split()
                id2vocab[int(id)] = char
        vocab = [0] * len(id2vocab)
        for id, char in id2vocab.items():
            vocab[id] = char
        return id2vocab, vocab

    def load_hotwords(self, hotwords_file):
        """
        load hotwords.yaml
        """
        with open(hotwords_file, 'r', encoding="utf-8") as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
        return configs

    def execute(self, requests):

        responses = []

        batch_encoder_lens = []
        batch_log_probs, batch_log_probs_idx = [], []
        batch_count = []
        batch_root = TrieVector()
        batch_start = []
        root_dict = {}

        encoder_max_len = 0
        hyps_max_len = 0
        total = 0
        for request in requests:
            # Perform inference on the request and append it to responses list...
            in_1 = pb_utils.get_input_tensor_by_name(request,
                                                     "encoder_out_lens")
            in_2 = pb_utils.get_input_tensor_by_name(request,
                                                     "batch_log_probs")
            in_3 = pb_utils.get_input_tensor_by_name(request,
                                                     "batch_log_probs_idx")

            cur_b_lens = in_1.as_numpy()
            # print('cur_b_lens: ', cur_b_lens)

            batch_encoder_lens.append(cur_b_lens)
            cur_batch = cur_b_lens.shape[0]
            batch_count.append(cur_batch)

            cur_b_log_probs = in_2.as_numpy()
            cur_b_log_probs_idx = in_3.as_numpy()

            # print('cur_b_log_probs: ', cur_b_log_probs)
            # print('cur_b_log_probs_idx: ', cur_b_log_probs_idx)
            for i in range(cur_batch):
                cur_len = cur_b_lens[i]
                cur_probs = cur_b_log_probs[i][
                    0:cur_len, :].tolist()  # T X Beam
                cur_idx = cur_b_log_probs_idx[i][
                    0:cur_len, :].tolist()  # T x Beam

                batch_log_probs.append(cur_probs)
                batch_log_probs_idx.append(cur_idx)
                root_dict[total] = PathTrie()
                batch_root.append(root_dict[total])
                batch_start.append(True)
                total += 1

        score_hyps = ctc_beam_search_decoder_batch(
            batch_log_probs,
            batch_log_probs_idx,
            batch_root,
            batch_start,
            self.beam_size,
            min(total, self.num_processes),
            blank_id=self.blank_id,
            space_id=-2,
            cutoff_prob=self.cutoff_prob,
            ext_scorer=self.lm,
            hotwords_scorer=self.hotwords_scorer)

        all_hyps = []
        all_ctc_score = []
        max_seq_len = 0
        for seq_cand in score_hyps:
            # if candidates less than beam size
            if len(seq_cand) != self.beam_size:
                seq_cand = list(seq_cand)
                seq_cand += (self.beam_size - len(seq_cand)) * [(-float("INF"),
                                                                 (0, ))]
            for score, hyps in seq_cand:
                all_hyps.append(list(hyps))
                all_ctc_score.append(score)
                max_seq_len = max(len(hyps), max_seq_len)

        beam_size = self.beam_size
        hyps_max_len = max_seq_len + 2
        in_ctc_score = np.zeros((total, beam_size), dtype=self.data_type)
        in_hyps_pad_sos_eos = np.ones(
            (total, beam_size, hyps_max_len), dtype=np.int64) * self.eos

        in_hyps_lens_sos = np.ones((total, beam_size), dtype=np.int32)

        st = 0
        for b in batch_count:
            for i in range(b):
                for j in range(beam_size):
                    cur_hyp = all_hyps.pop(0)
                    cur_len = len(cur_hyp) + 2
                    in_hyp = [self.sos] + cur_hyp + [self.eos]
                    in_hyps_pad_sos_eos[st + i][j][0:cur_len] = in_hyp
                    in_hyps_lens_sos[st + i][j] = cur_len - 1
                    in_ctc_score[st + i][j] = all_ctc_score.pop(0)
            st += b

        hyps = []
        idx = 0
        for cands, cand_lens in zip(in_hyps_pad_sos_eos, in_hyps_lens_sos):
            best_idx = 0
            best_cand_len = cand_lens[best_idx] - 1  # remove sos
            best_cand = cands[best_idx][1:1 + best_cand_len].tolist()
            hyps.append(best_cand)
            idx += 1

        hyps = map_batch(
            hyps, self.vocabulary,
            min(multiprocessing.cpu_count(), len(in_ctc_score)))

        st = 0
        for b in batch_count:
            for hyp in hyps[st:st + b]:
                if hyp:
                    # print('hyp: ', hyp)
                    seg_list = jieba.cut(hyp)
                    txt_res = " ".join(seg_list)
                    predict = self.use_punctuator.handle(txt_res)
                    res = utils.add_punc_to_txt(txt_res, predict, self.use_punctuator.class2punc)

            sents = np.array(hyps[st:st + b])
            out0 = pb_utils.Tensor("OUTPUT0",
                                   sents.astype(self.out0_dtype))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out0])
            responses.append(inference_response)
            st += b
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
