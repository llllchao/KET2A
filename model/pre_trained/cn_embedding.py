import torch
from modelscope.pipelines import pipeline
from modelscope.models import Model


def get_embeddings(entities: list):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 传入模型id或模型目录
    model_dir = r"/home/lichao/文档/KEA2T-fina/KEA2T-final/model/pre_trained/nlp_bert_entity-embedding_chinese-base"
    model = Model.from_pretrained(model_dir, task="sentence-embedding", device=device)
    pipeline_ee = pipeline('sentence-embedding', model, device=device)

    sentences = ["[ENT_S]%s[ENT_E]" % str(entity) for entity in entities]
    inputs = {
        "source_sentence": sentences
    }
    result = pipeline_ee(input=inputs)
    embeddings_list = result["text_embedding"]
    embeddings = torch.tensor(embeddings_list)
    return embeddings


if __name__ == "__main__":
    get_embeddings(["sb", "好人"])
