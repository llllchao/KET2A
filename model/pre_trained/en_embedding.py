import torch
from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


def get_embeddings(entities: list):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 传入模型id或模型目录
    model_dir = r"A:\projects\doing\KEA2T-final\model\pre_trained\nlp_corom_sentence-embedding_english-base"
    model = Model.from_pretrained(model_dir, task="sentence-embedding", device=device)
    pipeline_ee = pipeline('sentence-embedding', model, device=device)

    sentences = ["%s" % str(entity) for entity in entities]
    inputs = {
        "source_sentence": sentences
    }
    result = pipeline_ee(input=inputs)
    embeddings_list = result["text_embedding"]
    embeddings = torch.tensor(embeddings_list)
    return embeddings


if __name__ == "__main__":
    print(get_embeddings(["sb", "nihao"]))
