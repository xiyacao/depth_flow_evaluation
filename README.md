# depth_flow_evaluation

## Depth predictions for Zhou
First, download the pre-trained model by running the following
```bash
bash ./zhou/models/download_depth_model.sh
```
Then you can use the provided `./zhou/predictDepth.py` to get depth predictions. The result will be stored in `./zhou/output` as a npy file.  