import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from torch.optim import AdamW


# 1. Custom Dataset 정의
class MathTopicDataset(Dataset):
    def __init__(self, json_dir, label_mapping):
        self.data = []
        self.label_mapping = label_mapping

        # 데이터 로드
        for root, _, files in os.walk(json_dir):
            for file in files:
                if file.endswith(".json"):
                    filepath = os.path.join(root, file)
                    with open(filepath, "r", encoding="utf-8") as f:
                        item = json.load(f)
                        self.data.append(
                            {
                                "text": item["question_text"],
                                "label": label_mapping[item["question_topic_name"]],
                            }
                        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]["text"], self.data[idx]["label"]


# 2. 분류 모델 정의
class SentenceTransformerClassifier(nn.Module):
    def __init__(self, pretrained_model_name, num_labels):
        super(SentenceTransformerClassifier, self).__init__()
        self.encoder = SentenceTransformer(pretrained_model_name)
        self.classifier = nn.Linear(
            self.encoder.get_sentence_embedding_dimension(), num_labels
        )

    def forward(self, sentences):
        embeddings = self.encoder.encode(sentences, convert_to_tensor=True)
        logits = self.classifier(embeddings)
        return logits


# 3. 학습 코드
def main():
    # 설정
    json_dir = "./data/training"  # 학습 데이터 경로
    batch_size = 16
    num_epochs = 5
    learning_rate = 2e-5
    pretrained_model_name = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # 라벨 매핑 생성
    label_mapping = {}
    for root, _, files in os.walk(json_dir):
        for file in files:
            if file.endswith(".json"):
                filepath = os.path.join(root, file)
                with open(filepath, "r", encoding="utf-8") as f:
                    item = json.load(f)
                    label_mapping[item["question_topic_name"]] = label_mapping.get(
                        item["question_topic_name"], len(label_mapping)
                    )

    print(f"총 라벨 수: {len(label_mapping)}")

    # Dataset 및 DataLoader 초기화
    dataset = MathTopicDataset(json_dir, label_mapping)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 모델 초기화
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformerClassifier(pretrained_model_name, len(label_mapping)).to(
        device
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # 학습 루프
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for texts, labels in train_loader:
            labels = labels.to(device)
            logits = model(texts)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 검증 루프
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for texts, labels in val_loader:
                labels = labels.to(device)
                logits = model(texts)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # 로그 출력
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Train Loss: {train_loss / len(train_loader):.4f}, "
            f"Validation Loss: {val_loss / len(val_loader):.4f}, "
            f"Validation Accuracy: {100 * correct / total:.2f}%"
        )

    # 모델 저장
    torch.save(model.state_dict(), "sentence_transformer_classifier.pth")
    print("모델 학습 완료 및 저장됨.")


if __name__ == "__main__":
    main()
