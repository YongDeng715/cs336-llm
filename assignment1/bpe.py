# simple_bpe.py
from collections import Counter, defaultdict
import re

class BPETokenizer:
    def __init__(self, vocab=None, merges=None):
        # vocab: set of tokens (subword strings)
        # merges: list of merged pairs (tuples), in order
        self.vocab = vocab or set()
        self.merges = merges or []

    @staticmethod
    def get_stats(token_lists):
        """统计所有 token_lists (list of list of tokens) 中，
           相邻 pair 的出现频率"""
        pairs = Counter()
        for tokens in token_lists:
            for i in range(len(tokens)-1):
                pair = (tokens[i], tokens[i+1])
                pairs[pair] += 1
        return pairs

    @staticmethod
    def merge_pair(pair, token_lists):
        """在 token_lists 中，把所有 pair 出现位置合并为一个 token"""
        a, b = pair
        new_token = a + b
        new_token_lists = []
        for tokens in token_lists:
            i = 0
            new_tokens = []
            while i < len(tokens):
                # 若看到 a b，就合并
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i+1] == b:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_token_lists.append(new_tokens)
        return new_token_lists

    @classmethod
    def train(cls, corpus, target_vocab_size=1000, min_frequency=2):
        """
        corpus: list of strings (e.g. words or whitespace-separated text)
        target_vocab_size: 目标 vocab 大小 (包括基础 tokens + 合并后的 subwords)
        min_frequency: 合并对的最低频率 threshold — 避免过少次出现也被合并
        """
        # 初始：把每个 corpus 中 word 划分为 chars (并在词尾加特殊 end-of-word 标记)
        token_lists = []
        for line in corpus:
            # 也可以更复杂 pre-tokenize (word splitting), 这里只简化为按空格分词
            for word in line.strip().split():
                chars = list(word) + ['</w>']
                token_lists.append(chars)

        # 初始化 vocab: 所有基础字符 + </w>
        vocab = set(tok for tokens in token_lists for tok in tokens)
        merges = []

        while True:
            pairs = cls.get_stats(token_lists)
            if not pairs:
                break
            # 选出现频率最高的 pair
            most_common, freq = pairs.most_common(1)[0]
            if freq < min_frequency:
                print("Reached min_frequency with no more merges above threshold. Stopping.")
                break
            if len(vocab) >= target_vocab_size:
                print("Reached target vocab size. Stopping.")
                break

            # 合并
            token_lists = cls.merge_pair(most_common, token_lists)
            new_subword = most_common[0] + most_common[1]
            vocab.add(new_subword)
            merges.append(most_common)
            # 继续

        tokenizer = cls(vocab=vocab, merges=merges)
        return tokenizer

    def encode(self, text):
        """将输入 text 编码为 subword token list (不转 id，只 subword strings)"""
        tokens = []
        # 简单按空格切词
        for word in text.strip().split():
            # 初始 token 为每个字符 + </w>
            cur = list(word) + ['</w>']
            # 依序尝试所有 merge 规则 (按照训练时顺序)
            for a, b in self.merges:
                i = 0
                new = []
                while i < len(cur):
                    if i < len(cur)-1 and cur[i] == a and cur[i+1] == b:
                        new.append(a + b)
                        i += 2
                    else:
                        new.append(cur[i])
                        i += 1
                cur = new
            tokens.extend(cur)
        return tokens

    def decode(self, tokens):
        """把 subword tokens list decode 回字符串 (粗略，不一定完全恢复格式)"""
        words = []
        cur_word = ""
        for tok in tokens:
            if tok.endswith("</w>"):
                cur_word += tok[:-4]
                words.append(cur_word)
                cur_word = ""
            else:
                cur_word += tok
        # 若最后没有 </w>，也把 cur_word 加上
        if cur_word:
            words.append(cur_word)
        return " ".join(words)


if __name__ == "__main__":
    # 简单测试
    corpus = [
        "low lower lowest new newer wider", 
        "this is a test of BPE tokenizer", 
        "hello world hello world lower lowest"
    ]
    tokenizer = BPETokenizer.train(corpus, target_vocab_size=50, min_frequency=2)
    print("Vocab size:", len(tokenizer.vocab))
    print("Some merges:", tokenizer.merges[:10])

    for text in ["low", "lowest", "newest", "hello world", "lower low hello"]:
        toks = tokenizer.encode(text)
        print("Text:", text)
        print("  Tokens:", toks)
        print("  Decoded:", tokenizer.decode(toks))
        print()
