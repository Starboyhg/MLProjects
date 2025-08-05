# MLProjects
Here I will archive all my most relevant projects related to machine learning and artificial intelligence that I have worked on college and personal too.
#  LLM Benchmarking & Fine-Tuning for Text Classification

This project explores the performance of different open-source LLMs on a text classification task. It includes benchmarking, evaluation, and fine-tuning of models to better understand their trade-offs in terms of accuracy, inference speed, and resource usage.

> Designed as a project to gain hands-on experience with LLMs, pipelines, and model experimentation.

---

##  Objectives

- Evaluate and compare pre-trained LLMs on a downstream classification task
- Construct a testing ML pipeline to measure performance metrics
- Fine-tune one of the models for improved performance
- Document trade-offs between models for different deployment scenarios

---

## Dataset

- **Name**: [IMDb](https://huggingface.co/datasets/imdb) / [AG News](https://huggingface.co/datasets/ag_news) / [Emotion Dataset](https://huggingface.co/datasets/dair-ai/emotion)
- **Type**: Text classification
- **Classes**: `e.g., Positive / Negative`, `News categories`, etc.
- **Size**: ~25,000 training samples

---

## Models Evaluated

| Model                  | Params     | Size      | Inference Time | Accuracy |
|------------------------|------------|-----------|----------------|----------|
| `bert-base-uncased`    | 110M       | ~420MB    | TBD            | TBD      |
| `distilbert-base-uncased` | 66M    | ~255MB    | TBD            | TBD      |
| (Optional) Fine-tuned Model | N/A | N/A       | TBD            | TBD      |

> Metrics measured: Accuracy, F1-score, inference time per batch, memory usage

---

## Tech Stack

- Python
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [Datasets](https://huggingface.co/docs/datasets/index)
- scikit-learn
- Jupyter Notebooks
- Colab + GPU for fine-tuning
- MLflow for tracking experiments

---

## Results & Insights

> Findings

- 

## References

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [Fine-tuning Tutorial](https://huggingface.co/transformers/training)

---

## Author

**Jesús Pérez**  
Computer Systems Engineering Student |   
[LinkedIn](https://www.linkedin.com/in/perezgonzalezjesus1309) • [GitHub](https://github.com/Starboyhg)


---

## Project Structure

