# Dataset List

- annotated_data_{name} : The dataset of 150 examples annotated by {name}.
    - annotated_data_yifei : First trial by yifei.
    - annotated_data_yifei_v2 : Second trial by yifei.
    - annotated_data_llama2 : Few-shot inference by question answering without using prompt template.
    - annotated_data_llama2_v2 : Few-shot inference by question answering with prompt template.
    - annotated_data_llama2_v2_context : Few-shot inference by question answering with prompt template, label definitions.
- daccord_{name1}_{name2} : The dataset intersected cross datasets from {name1} and {name2} by labels.
- to_annotate_150_cleaned : The extracted 150 data to annotate.