import numpy as np
import torch, sys
from torchvision.transforms import functional as func
import torchvision.transforms as transforms
from loss import ComputeLoss
import yaml, random
import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
from new_model import Model
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from torch.utils.data import DataLoader

from dataloader import FiftyOneTorchDataset
from util import non_max_suppression

dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    # label_types=[""],
    classes=["cat", "dog"],
    max_samples=128,
)

dataset.persistent = True
view = dataset.filter_labels("ground_truth", F("label").is_in(("cat", "dog")))
predictions_view = view.take(128, seed=63)

fil_classes = ["cat", "dog"]
device = torch.device('cuda:0')

with open("hyp.yaml", "r") as stream:
    try:
        hyp = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

model = Model('yolov3.cfg', hyp=hyp).to(device)
optimizer = torch.optim.Adam(model.parameters(),1e-3)

org_w = 640
org_h = 480
scaling_factor = 640/480

batch_size = 8

transform = transforms.Compose([transforms.Resize((int(org_h/scaling_factor), int(org_w/scaling_factor))),
                                transforms.Pad((0, int((org_w - org_h)/(2*scaling_factor)),0,int((org_w - org_h)/(2*scaling_factor)))),
                                transforms.ToTensor()])
            
dataset = FiftyOneTorchDataset(predictions_view, transform, classes=fil_classes)
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

loss_fcn = ComputeLoss(model)
# Add predictions to samples\
epochs = 1000
for epoch in range(epochs):
    tot_loss = 0
    count = 0
    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)
        preds = model(images)
        loss, loss_parts = loss_fcn(preds, targets)
        tot_loss += loss / batch_size
        count += 1
        loss.backward()
        optimizer.step()
    print(epoch, tot_loss.item()/count)


#  TODO: val code, val dataloader, save model, save training loss

print(sample.ground_truth.detections)
model.eval()
pred = model(image)
print(pred[0,0])
detections = non_max_suppression(pred, conf_thres=0.7, iou_thres=0.1,max_det=300)[0].cpu().detach().numpy()


img = image[0,:].cpu().detach().numpy().transpose(1,2,0)
plt.figure()
fig, ax = plt.subplots(1)
ax.imshow(img_org)
# Rescale boxes to original image
unique_labels = np.unique(detections[:, -1])
n_cls_preds = len(unique_labels)
# Bounding-box colors
cmap = plt.get_cmap("tab20b")
colors = [cmap(i) for i in np.linspace(0, 1, n_cls_preds)]
bbox_colors = random.sample(colors, n_cls_preds)
for x1, y1, x2, y2, conf, cls_pred in detections:

    print(f"\t+ Label: {fil_classes[int(cls_pred)]} | Confidence: {conf.item():0.4f}")
    
    x1, y1, x2, y2 = x1*scaling_factor, y1*scaling_factor, x2*scaling_factor, y2*scaling_factor

    box_w = (x2 - x1)
    box_h = (y2 - y1)
    color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
    # Create a Rectangle patch
    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
    # Add the bbox to the plot
    ax.add_patch(bbox)
    # Add label
    plt.text(
        x1,
        y1,
        s=fil_classes[int(cls_pred)],
        color="white",
        verticalalignment="top",
        bbox={"color": color, "pad": 0})

# Save generated image with detections
plt.axis("off")
plt.gca().xaxis.set_major_locator(NullLocator())
plt.gca().yaxis.set_major_locator(NullLocator())
plt.show()