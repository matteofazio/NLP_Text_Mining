# -*- coding: utf-8 -*-
"""
Utility module for expertai-nlpy.

@author: Andrea Belli <abelli@expert.ai>
"""


from tqdm import tqdm, trange


def tokens_to_docs(raw, eai):
    """Analyze a sentence with expertai
    
    Take a list of sentences, where each sentence is a list of token; build a
    string with the sentence and analyze it with expertai.
    
    Params:
        raw: list of lists of tokens
        eai: Expertai instance
    Return:
        docs: list of expertai Document
    """
    docs = []
    for sent in tqdm(raw):
        docs.append(eai.analyze(' '.join(sent), {'features': ['dependency', 'knowledge']}))
    return docs


def tokens_to_docs_safe(raw, eai):
    """Analyze a sentence with expertai
    
    Take a list of sentences, where each sentence is a list of token; build a
    string with the sentence and analyze it with expertai. This function 
    differs from token_to_docs because this deals the case of exceptions in
    the Expertai.analyze method.
    
    Params:
        raw: list of lists of tokens
        eai: Expertai instance
    Return:
        docs: list of expertai Document
        errors: list of indexes of sentences wich raised an exception
    """
    errors = []
    docs = []
    for idx, sent in tqdm(enumerate(raw), total=len(raw)):
        try:
            docs.append(eai.analyze(' '.join(sent), {'features': ['dependency', 'knowledge']}))
        except:
            errors.append(idx)
            
    print(len(errors), 'errors')
    return docs, errors


def _voidtoken():
    """Generate an empty token"""
    t = {
        'word': '',
        'pos': '',
        'syncon': -1,
        'ancestor': -1,
        'dep': '',
        'label': ''
    }
    return t


def _get_ancestor(syncon, eai):
    """Given a syncon, return (one of) his ancestor(s)"""
    if syncon == -1:
        return -1
    
    params = {"syncon": syncon,
              "link_name": "supernomen/subnomen", 
              "direction": "from", "level": 1}
    ancestor = eai.kgraph.linked_syncons(params, 0, 1)
    if ancestor.max_records == 0:
        return -1
    else:
        return ancestor.syncon_list[0]

    
def _get_label(doc, syncon):
    """Extract the knowledge label of a syncon in a document, if any"""
    label = ''
    if hasattr(doc.knowledge, '_k'):
        for ent in doc.knowledge._k:
            if ent['syncon'] == syncon:
                label = ent['label']
                break
        if label and '.' in label:
            label = label.split('.')[-1]
    return label
    

def nlpy_features(sentences, docs, eai):
    """Extract token features from expertai docs
    
    Given a list of tokenized sentences and the relative expertai docs, 
    create a dictionary for each with the doc features:
        * Word
        * PoS tag;
        * Dep tag;
        * Syncon;
        * Ancestor syncon;
        * Label;
        * Typeclass (a mix of POS and entity);
    Params:
        sentences: list of sentences, that are lists of strings;
        docs: list of expertai Document;
        eai: Expertai instance;
    Returns:
        eai_sents: list of sentences features, that are lists of dictionaries;
    """
    eai_sents = []
    for sent_idx in trange(len(sentences)):
        seek = 0    # Index of the part of the sentence string already read
        eai_tokenlist = []
        for tk_idx in range(len(sentences[sent_idx])):
            # Token text and boundary indexes in doc.content
            token = sentences[sent_idx][tk_idx]
            index_start = docs[sent_idx].content.find(token, seek)
            index_end = index_start + len(token)
            possible_tokens = []
            for t in docs[sent_idx].tokens:
                # If a eai Token contain (part of the) chunk od text, it can be
                # the possible corresponding Token
                if (t.start<=index_start and t.end>=index_end) or \
                (t.start >= index_start and t.start <= index_end) or \
                (t.end >= index_start and t.end <= index_end):
                    possible_tokens.append(t)
            if not possible_tokens:
                print('ERROR: expertai tokenization not found for token', token)
                eai_tokenlist.append(_voidtoken())
            else:
                # Extract information from the eai.Token for the raw token we 
                # are analyzing
                if len(possible_tokens)>1:
                    possible_tokens.sort(key = lambda t: t.syncon, reverse=True)
                new_token = {
                    'word': token,
                    'pos': possible_tokens[0].pos,
                    'syncon': possible_tokens[0].syncon,
                    'ancestor': _get_ancestor(possible_tokens[0].syncon, eai),
                    'label': _get_label(docs[sent_idx], possible_tokens[0].syncon),
                    'dep': possible_tokens[0].dependency.label,
                    'typeclass': possible_tokens[0].typeclass.split('.')
                }
                eai_tokenlist.append(new_token)
            seek = index_end
            while docs[sent_idx].content[seek] == ' ':
                seek += 1
        eai_sents.append(eai_tokenlist)
    return eai_sents


def _nlpy_word_features(sentence, idx):
    """Extract features related to a word and its neighbours"""
    token = sentence[idx] 
    
    features = {
        'bias': 1.0,
        'word.lower()': token['word'].lower(),
        'word[-3:]': token['word'][-3:],
        'word[-2:]': token['word'][-2:],
        'word.isupper()': token['word'].isupper(),
        'word.istitle()': token['word'].istitle(),
        'word.isdigit()': token['word'].isdigit(),
        'nlpy.postag': token['pos'],
        'nlpy.postag[:2]': token['pos'][:2],
        'nlpy.deptag': token['dep'],
        'nlpy.deptag[-2:]': token['dep'][-2:],
        'nlpy.syncon': -1 if token['syncon'] == -1 else token['syncon'] / 10000.,
        'nlpy.ancestor': -1 if token['ancestor'] == -1 else token['ancestor'] / 10000.,
        'nlpy.labels': token['label'],
        'nlpy.typeclass': token['typeclass'],
    }
    if idx > 0:
        token1 = sentence[idx-1]
        features.update({
            '-1:word.lower()': token1['word'].lower(),
            '-1:word.istitle()': token1['word'].istitle(),
            '-1:word.isupper()': token1['word'].isupper(),
            '-1:nlpy.postag': token1['pos'],
#             '-1:nlpy.postag[:2]': token1['pos'][:2],
            '-1:nlpy.deptag': token1['dep'],
#             '-1:nlpy.deptag[-2:]': token1['dep'][-2:],
#             '-1:nlpy.syncon': -1 if token1['syncon'] == -1 else token1['syncon'] / 10000.,
#             '-1:nlpy.ancestor': -1 if token1['ancestor'] == -1 else token1['ancestor'] / 10000.,
            '-1:nlpy.labels': token1['label'],
            '-1:nlpy.typeclass': token1['typeclass'],
        })
    else:
        features['BOS'] = True
        
    if idx < len(sentence)-1:
        token1 = sentence[idx-1]
        features.update({
            '+1:word.lower()': token1['word'].lower(),
            '+1:word.istitle()': token1['word'].istitle(),
            '+1:word.isupper()': token1['word'].isupper(),
            '+1:nlpy.postag': token1['pos'],
#             '+1:nlpy.postag[:2]': token1['pos'][:2],
            '+1:nlpy.deptag': token1['dep'],
#             '+1:nlpy.deptag[-2:]': token1['dep'][-2:],
#             '+1:nlpy.syncon': -1 if token1['syncon'] == -1 else token1['syncon'] / 10000.,
#             '+1:nlpy.ancestor': -1 if token1['ancestor'] == -1 else token1['ancestor'] / 10000.,
            '+1:nlpy.labels': token1['label'],
            '+1:nlpy.typeclass': token1['typeclass'],
        })
    else:
        features['EOS'] = True
                
    return features


def nlpy_sentence_features(sentence):
    """Create feature dictionary for a sentence"""
    return tuple(_nlpy_word_features(sentence, index) for index in range(len(sentence)))