"""Writes a file of incorrect predictions, and the list of trigrams containing errors.

Given model predictions and gold outputs, we get alignments between incorrect preds and gold
and then write those trigrams containing unaligned character(s)."""

import argparse
from typing import List
from collections import Counter

import pynini
from pynini.lib import edit_transducer

UPDATE_SOURCE_ACTIONS = set(["D", "S"])
UPDATE_TARGET_ACTION = "I"

class Aligner:
    def __init__(self, alphabet):
        # alphabet = set.union(*[set(word + seg) for word, seg in read(segments_file)])
        with pynini.default_token_type("utf8"):
            self.alphabet = set(alphabet)
            self.automaton = edit_transducer.EditTransducer((pynini.escape(ch) for ch in alphabet))

    def __call__(self, pred: str, gold: str):
        with pynini.default_token_type("utf8"):
            try:
                lattice = self.automaton.lattice(
                    pynini.escape(pred),
                    pynini.escape(gold),
                )
            except Exception as e:
                print(pred, gold, )
                raise e
            path = pynini.shortestpath(lattice)
            paths = path.paths()
            labeled_pred = []
            pred = ""
            gold = ""
            for i, o in zip(paths.ilabels(), paths.olabels()):
                labeled_pred.append(self.make_label(i, o))
                pred += self._dec(i)
                gold += self._dec(o)
            return pred, labeled_pred, gold

    def _dec(self, i: int):
        return chr(i)

    def make_label(self, i: int, o: int) -> str:
        if i == 0:
            return "I"
        elif o == 0:
            return "D"
        elif i != o:
            return "S"
        else:
            return "_"


def get_trigrams(p: str, i: int) -> List[str]:
    """Gets all trigrams in p involving the position i

    Args:
        p (str): _description_
        i (int): _description_

    Returns:
        List[str]: _description_
    """
    # One-liner for trigrams, where we get all trigrams in the 
    # substring from i-2 to i+3.
    trigrams =  list(zip(*[p[max(i-2, 0): i+3][j:] for j in range(3)]))
    return trigrams
    

def get_pred_ngrams(pred, action, gold):
    last_is_action = False
    ngrams = []
    ctx_trgrams = []
    for i, (p, a) in enumerate(zip(pred, action)):
        # If the action is on the existing source string
        # then we track src ngrams
        if a in UPDATE_SOURCE_ACTIONS:
            # Removes the \x00's that are there to keep alignment.
            ctx_trgrams.extend(get_trigrams([p for p in pred if p != "\x00"], i))
            if last_is_action:
                ngrams[-1] += p
            else:
                ngrams.append(p)
                last_is_action = True
        # If the action entails inserting a char
        # then we have no source ngram.
        elif a == UPDATE_TARGET_ACTION:
            continue
        else:
            last_is_action = False

    return ngrams, ctx_trgrams


def get_target_ngrams(pred, action, gold):
    last_is_action = False
    ngrams = []
    ctx_trgrams = []
    for i, (g, a) in enumerate(zip(gold, action)):
        # If the action is on the existing source string
        # then we track src ngrams
        if a == UPDATE_TARGET_ACTION:
            # Removes the \x00's that are there to keep alignment.
            ctx_trgrams.extend(get_trigrams([g for g in gold if g != "\x00"], i))
            if last_is_action:
                ngrams[-1] += g
            else:
                ngrams.append(g)
                last_is_action = True
        # If the action entails deleting or substituting a char
        # then we have no target ngram.
        elif a in UPDATE_SOURCE_ACTIONS:
            continue
        else:
            last_is_action = False

    return ngrams, ctx_trgrams


def get_exact_target_ngrams(pred, action, gold):
    pass


def get_context_pred_ngrams(pred, action, gold):
    pass


def get_context_target_ngrams(pred, action, gold):
    pass


def read_gold(filename: str):
    """Reads the gold file"""
    data = []
    with open(filename,"r") as f:
        for line in f:
            # Assumes 2023 format
            lemma, msd, form = line.rstrip().split("\t")
            data.append((lemma, msd, form))
    return data


def read_pred(filename: str):
    """Reads the predictions file"""
    data = []
    with open(filename,"r") as f:
        for line in f:
            data.append(line.rstrip())
    return data


def get_error_trigrams(aligner: Aligner, pred: str, gold: str) -> List[str]:
    """Gets the predicted trigrams containing an error
    wrt gold.

    Args:
        pred (str): Predicted word.
        gold (str): Gold word.

    Returns:
        List[str]: A list of predicted trigrams containing errors.
    """
    # 1. Get an alignment from pred to gold
    pred, action, gold = aligner(pred, gold)
    # 2. 
    exact_pred_ngrams, pred_trigram_contexts = get_pred_ngrams(pred, action, gold)
    exact_tgt_ngrams, tgt_trigram_contexts = get_target_ngrams(pred, action, gold)
    return (
        exact_pred_ngrams,
        exact_tgt_ngrams,
        tgt_trigram_contexts,
        pred_trigram_contexts,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred_fn",
        required=True,
    )
    parser.add_argument(
        "--gold_fn",
        required=True,
    )
    args = parser.parse_args()
    # Read data
    preds = read_pred(args.pred_fn)
    golds = read_gold(args.gold_fn)
    # Make transducer alphabet
    alphabet = set.union(*[set(word) for word in preds])
    alphabet = alphabet.union(*[set(word) for _, _, word in golds])
    aligner = Aligner(alphabet)
    # get_error_trigrams(aligner, "weingeweissten", "wiesen hin")
    # Compare each.
    exact_pred_ngrams = []
    exact_target_ngrams = []
    context_pred_ngrams = []
    context_target_ngrams = []
    for pred, gold_tup in zip(preds, golds):
        _, _, gold = gold_tup
        if pred != gold:
            ep, et, tt, pt, = get_error_trigrams(aligner, pred, gold)
            exact_pred_ngrams.extend(ep)
            exact_target_ngrams.extend(et)
            context_pred_ngrams.extend(tt)
            context_target_ngrams.extend(pt)

    print("EXACT PRED NGRAMS")
    print(Counter(exact_pred_ngrams))
    print("EXACT TGT NGRAMS")
    print(Counter(exact_target_ngrams))
    print("CONTEXT PRED TRIGRAMS")
    print(Counter(context_pred_ngrams))
    print("CONTEXT TGT TRIGRAMS")
    print(Counter(context_target_ngrams))



if __name__ == "__main__":
    main()