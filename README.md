# An Exploration of Text Infilling Techniques and Custom Mask Design


## Running the Code
First, upload the downloaded folder `Project` into Google Drive, open the notebook `project.ipynb` in Colab. <br> 
The main file `project.ipynb` is split into three sections: 

1. Introduction and setup
2. Training
3. Inference

Please execute the sections in order.
If you only want to see the inference examples, skip the training section.

Steps and guidelines are also provided in `project.ipynb`.

## Code Base

The implementation is extended from the [repository](https://github.com/chrisdonahue/ilm) provided in (Donahue et al., 2020). The code base fine-tunes a language model to infill blanks.

## Code Description

### Original files created by student

The main file created for execution is `project.ipynb`. This file executes several tasks: 
1. Clone repository provided from (Donahue et al., 2020).
2. Set up environment and install requirements.
3. Prepare and process the ROC stories dataset using `ilm/create_ilm_examples.py` and the custom mask created in `ilm/ilm/mask/custom.py`.
4. Train a new model with the infilling examples created in the previous step using `ilm/train_ilm.py`.
5. Perform infilling task with the new model.
6. Perform infilling task with the trained model from the
ILM paper.
7. Compare the results.

### Modified files from code base
- File: `ilm/ilm/mask/custom.py` <br>
    Description: This code defines a masking function (MaskCommonNoun) for common nouns in a given document. The mask function takes a document as input, tokenizes it, and performs part-of-speech tagging. It then iterates through the tokens, checking if they are common nouns ('NN' for singular and 'NNS' for plural) based on NLTK's part-of-speech tags. If a token is a common noun and a randomly generated number is less than the masking probability (p), it adds the span's information (type, starting offset, length) to the list of masked spans.

    ```
    # Modified

    class MaskCommonNounType(Enum):
        COMMON_NOUN = 0

    class MaskCommonNoun(MaskFn):
        def __init__(self, p=1.):
            try:
                pos_tag(['Ensure', 'tagger'])
            except:
                raise ValueError('Need to call nltk.download(\'averaged_perceptron_tagger\')')
            self.p = p

        @classmethod
        def mask_types(cls):
            return list(MaskCommonNounType)

        @classmethod
        def mask_type_serialize(cls, m_type):
            return m_type.name.lower()

        def mask(self, doc):
            masked_spans = []
            toks = word_tokenize(doc)
            toks_offsets = tokens_offsets(doc, toks)
            toks_pos = pos_tag(toks)
            for t, off, (_, pos) in zip(toks, toks_offsets, toks_pos):
                if pos in ['NN', 'NNS'] and random.random() < self.p:
                    masked_spans.append((MaskCommonNounType.COMMON_NOUN, off, len(t)))
            return masked_spans
    ```
- File: `ilm/train_ilm.py` <br>
    Description: To fix the TypeError that occured when using the original code from the repository, changes were made on how the logits of the model were received. 

    ```
    # Original

    logits, _ = model(inputs)
    eval_logits, _ = model(eval_inputs)

    # Modified

    logits = model(inputs).logits
    eval_logits = model(eval_inputs).logits
    ```

- File: `ilm/create_ilm_examples.py` <br>
    Description: To reduce the time to process the dataset, I added the following to allow users to specify a fraction of dataset to process.

    ```
    # Modified

    parser.add_argument('--subsample_fraction', type=float, default=1.0, help='Fraction of the dataset to process')

    num_documents_to_process = int(len(docs) * args.subsample_fraction)
    docs = random.sample(docs, num_documents_to_process)
    ```



## Datasets

The trained model provided in the paper was originally trained on three datasets: 

- Abstracts (200K examples, 30M words): Abstracts from CS papers on arXiv. 
- Lyrics (2M examples, 60M words): Song lyrics from lyrics.com.
- Stories (100K examples, 5M words): Short stories that contain a title and five sentences from the [ROC stories](https://cs.rochester.edu/nlp/rocstories/) dataset (Mostafazadeh et al., 2016).

Due to computation and time limits, only one-tenth of the ROC stories dataset is processed for the retrained model with custom mask.