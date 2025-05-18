"""
AI训练进度上报示例代码
将此代码片段集成到你的训练脚本中，在每个epoch或batch结束时调用report_progress_to_server函数。
"""
import requests
import time
import uuid
import json
import os

# 后端API地址
API_ENDPOINT = os.getenv("MONITORING_API_ENDPOINT", "")
# 本次训练的唯一ID（建议每次训练唯一）
TRAINING_ID = 'MambaTranslate'

def report_progress_to_server(epoch, loss, accuracy=None, batch=None, total_batches=None, lr=None, custom_metrics=None):
    """向监控服务器上报训练进度"""
    payload = {
        "training_id": TRAINING_ID,
        "epoch": int(epoch),
        "loss": float(loss),
        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    
    if batch is not None:
        payload["batch"] = int(batch)
    if total_batches is not None:
        payload["total_batches"] = int(total_batches)
    if accuracy is not None:
        payload["accuracy"] = float(accuracy)
    if lr is not None:
        payload["learning_rate"] = float(lr)
    if custom_metrics:
        payload["custom_metrics"] = custom_metrics

    try:
        headers = {'Content-Type': 'application/json'}
        response = requests.post(API_ENDPOINT, data=json.dumps(payload), headers=headers, timeout=5)
        response.raise_for_status()
        #print(f"[上报成功] epoch={epoch}, batch={batch if batch else '-'} status={response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"[上报失败] {e}")
