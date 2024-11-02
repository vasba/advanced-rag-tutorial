"""
1. Proposition extraction: The text is broken down into smaller, self-contained statements called propositions using techniques like dependency parsing. These propositions capture the key points of the text.

2. LLM evaluation: Each proposition is fed to an LLM trained on conversation data. This step leverages the LLM’s understanding of language to evaluate the propositions. The LLM’s training on diverse datasets helps it understand nuances, context, and relevance, enabling it to make informed decisions about the propositions.

3. Contextual understanding: The LLM, based on its training, determines whether the proposition:

    a. Belongs to an existing chunk because it’s relevant to the current topic.

    b. Requires a new chunk due to a significant shift in meaning (e.g., a new topic introduced).



"""