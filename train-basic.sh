source venv/bin/activate
flow --model cfg/darkflow-airplane-train.cfg --load -1 --train --annotation data/coco/annotations/airplane --dataset data/coco/images/airplane --backup checkpoint-airplane --gpu 0.9 --savepb



