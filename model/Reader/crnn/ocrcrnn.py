import torch.nn as nn

class OCRCRNN(nn.Module):
    def __init__(self,NUM_CLASSE=11):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),      # 20x50 -> 10x25

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),      # 10x25 -> 5x12
        )

        self.rnn = nn.LSTM(
            input_size=128 * 5,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Linear(256, NUM_CLASSE)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(b, w, c * h)

        x, _ = self.rnn(x)
        x = self.fc(x)
        return x
    
class OCRCRNNgray(nn.Module):
    def __init__(self,NUM_CLASSES=11,CHANNELS=1):
        super().__init__()

        self.cnn = nn.Sequential(
            # Lớp đầu vào đổi từ 3 thành 1
            nn.Conv2d(CHANNELS, 64, 3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2),      # 20x50 -> 10x25

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),      # 10x25 -> 5x12
        )

        self.rnn = nn.LSTM(
            input_size=128 * 5,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Linear(256, NUM_CLASSES)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(b, w, c * h)

        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

class TinyCRNN(nn.Module):
    def __init__(self, CHANNELS=1, NUM_CLASSES=11):
        super().__init__()
        # Giảm số filter của CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(CHANNELS, 32, 3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2),      # 20x50 -> 10x25

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),      # 10x25 -> 5x12
        )

        # Sử dụng GRU và giảm hidden size
        self.rnn = nn.GRU(
            input_size=64 * 5,
            hidden_size=64,
            num_layers=1, # Giảm xuống 1 lớp
            bidirectional=True,
            batch_first=True
        )

        # FC Layer nhỏ hơn (64*2 = 128)
        self.fc = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(b, w, c * h)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x