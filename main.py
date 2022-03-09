from darknet import Darknet
import numpy as np
import torch, sys
from PIL import Image
from torchvision.transforms import functional as func
import torchvision.transforms as transforms
from loss import ComputeLoss
import yaml, random
import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

from util import non_max_suppression

dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    # label_types=[""],
    classes=["cat", "dog"],
    max_samples=25,
)

dataset.persistent = True
view = dataset.filter_labels("ground_truth", F("label").is_in(("cat", "dog")))

predictions_view = view.take(1, seed=63)

# Get class list
# classes = dataset.default_classes
# fil_classes = []
# for class_idx in classes:
#     if not class_idx.isnumeric() and class_idx in ["cat", "dog"]:
#         fil_classes.append(class_idx)
# sys.exit()
# print(len(fil_classes))
fil_classes = ["cat", "dog"]
device = torch.device('cuda:0')

with open("hyp.yaml", "r") as stream:
    try:
        hyp = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

model = Darknet('yolov3.cfg', hyp).to(device)

optimizer = torch.optim.Adam(model.parameters(),1e-3)

org_w = 640
org_h = 480

scaling_factor = 640/480


loss_fcn = ComputeLoss(model)
# Add predictions to samples\
epochs = 100
for epoch in range(epochs):
    tot_loss = 0
    count = 0
    with fo.ProgressBar() as pb:
        for sample in pb(predictions_view):
            model.train()
            optimizer.zero_grad()
            # print(sample.ground_truth.detections)
            # create targets [img_id, class_idx, bboxes(x, y, w, h)]
            target = sample.ground_truth.detections[0]
            cur_class = fil_classes.index(target.label)
            bbox = target.bounding_box
            # print(bbox)
            bbox[0] = bbox[0] + bbox[2]/2
            bbox[1] = bbox[1] + bbox[3]/2
            tmp = np.append(0, cur_class)
            tar = np.append(tmp, bbox)
            targets = torch.tensor(tar)[None,:]
            for target in sample.ground_truth.detections[1:]:
                cur_class = fil_classes.index(target.label)
                bbox = target.bounding_box
                bbox[0] = bbox[0] + bbox[2]/2
                bbox[1] = bbox[1] + bbox[3]/2
                tmp = np.append(0, cur_class)
                tar = np.append(tmp, bbox)
                curr_target = torch.tensor(tar)[None,:]
                targets = torch.vstack((targets, curr_target))

            targets = targets.to(device)
            img_org = Image.open(sample.filepath)
            height, width = img_org.size
            transform = transforms.Compose([transforms.Resize((int(org_h/scaling_factor), int(org_w/scaling_factor))),
                                            transforms.Pad((0, int((org_w - org_h)/(2*scaling_factor)),0,int((org_w - org_h)/(2*scaling_factor))))])
            image = transform(img_org)
            image = func.to_tensor(image).to(device)
            c, h, w = image.shape
            image = image[None,:]
            # Perform inference
            preds = model(image)
            # calc loss
            loss, loss_parts = loss_fcn(preds, targets)
            tot_loss += loss
            count += 1
            loss.backward()
            optimizer.step()
        print(epoch, tot_loss.item()/count)

        # labels = preds["labels"].cpu().detach().numpy()
        # scores = preds["scores"].cpu().detach().numpy()
        # boxes = preds["boxes"].cpu().detach().numpy()
        # # Convert detections to FiftyOne format
        # detections = []
        # for label, score, box in zip(labels, scores, boxes):
        #     # Convert to [top-left-x, top-left-y, width, height]
        #     # in relative coordinates in [0, 1] x [0, 1]
        #     x1, y1, x2, y2 = box
        #     rel_box = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]

        #     detections.append(
        #         fo.Detection(
        #             label=classes[label],
        #             bounding_box=rel_box,
        #             confidence=score
        #         )
        #     )

        # # Save predictions to dataset
        # sample["faster_rcnn"] = fo.Detections(detections=detections)
        # sample.save()

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