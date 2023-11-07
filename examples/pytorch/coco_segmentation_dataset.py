
import torch
import json
from pathlib import Path
from PIL import Image
from pycocotools.coco import COCO
import numpy as np

class CocoSegmentation(torch.utils.data.Dataset):
    def __init__(self, img_folder, ann_file):
        self.coco = COCO(ann_file)
        self.img_folder = Path(img_folder)
        self.cat_ids = self.coco.getCatIds()

        self.id_to_annId = {}
        for seq_id, (annot_id, annot) in enumerate(self.coco.anns.items()):
            image_id = annot["image_id"]
            category_id = annot["category_id"]
            self.id_to_annId[seq_id] = {"annot_id": annot_id, "img_id": image_id, "file_name": self.coco.imgs[image_id]["file_name"], "category_id": category_id}


    def __len__(self):
        return len(self.coco.anns)
    
    def __getitem__(self, idx):
        annot = self.id_to_annId[idx]
        
        img_path = self.img_folder / annot["file_name"]
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        
        anns = self.coco.loadAnns(annot["annot_id"])
        mask = self.coco.annToMask(anns[0])
        
        category_id = annot["category_id"]
        category_name = self.coco.cats[category_id]["name"]

        masked_img = img.copy()
        for channel in range(masked_img.shape[-1]):
            masked_img[:,:,channel] *= mask
            
        return {"image": img, "mask": mask, "category": category_name, "masked_img": masked_img}
