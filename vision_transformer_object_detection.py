import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward):
        super(TransformerBlock, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = self.layer_norm(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.layer_norm(x + self.dropout(ff_output))
        return x

class TransformerObjectDetector(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_classes):
        super(TransformerObjectDetector, self).__init__()
        self.transformer_block = TransformerBlock(d_model, nhead, dim_feedforward)
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        x = self.transformer_block(x)
        x = self.classifier(x)
        return x

model = TransformerObjectDetector(512, 8, 2048, 80)

#Data Preprocesssing
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pycocotools.coco as coco


# Define the transformations to apply to the images
transform = transforms.Compose([
    transforms.Resize(512),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class COCODataset(Dataset):
    def __init__(self, root_dir, set_name='train2017'):
        self.coco = coco.COCO(f'{root_dir}/annotations/instances_{set_name}.json')
        self.image_ids = self.coco.getImgIds()
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image = self.coco.loadImgs(image_id)[0]
        image_data = ... # Load the image data
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id))
        targets = ... # Process the annotations
        return image_data, targets

# train_dataset = COCODataset(root_dir='./train2017')
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=lambda x: x)
# Load the COCO train2017 dataset
train_dataset = datasets.CocoDetection(root='./train2017', annFile='./annotations/instances_train2017.json', transform=transform)

# Create a data loader to load the data in batches
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=lambda x: x)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# Train the model for a few epochs

for epoch in range(10):
    # Loop over the training data
    for batch_idx,(input,output) in enumerate(train_loader):
        # input = batch[0][0]
        # label = batch[0][1]
        optimizer.zero_grad()
        outputs = model(input)
        loss = criterion(outputs,label)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')