"""services — lớp xử lý logic, mỗi file tương ứng 1 nhóm API/việc.

Đặt tên theo API/việc; gọi vào folder gốc tác giả (qua ``src.engine``) hoặc ra
provider ngoài (Vast.ai). Controller mỏng, đẩy hết logic xuống đây.

* ``createVoice`` — orchestrator API tạo TTS: check điều kiện + rẽ nhánh CPU/GPU.
* ``cpu_onnx``    — chạy synth in-process trên máy này (CPU/ONNX).
* ``gpu_vastai``  — provision Vast.ai on-demand (tạo→chạy→upload→destroy). (skeleton)
* ``vastai``      — client gọi API Vast.ai (chừa cho user tự hoàn thiện).
* ``voices``      — CRUD giọng (clone).
* ``catalog``     — styles / modes / health.
* ``jobs``        — quản lý job async (in-memory).
"""
