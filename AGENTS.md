# AGENTS.md

Các nguyên tắc hành vi giúp giảm thiểu những lỗi code thường gặp của LLM. Có thể gộp chung với các hướng dẫn cụ thể của dự án khi cần.

**Sự đánh đổi:** Các nguyên tắc này thiên về sự cẩn trọng hơn là tốc độ. Đối với các tác vụ đơn giản, hãy tự cân nhắc.

## 1. Suy Nghĩ Trước Khi Code

**Không tự đưa ra giả định. Không giấu giếm sự bối rối. Trình bày rõ các sự đánh đổi.**

Trước khi triển khai:
- Trình bày các giả định của bạn một cách rõ ràng. Nếu không chắc chắn, hãy hỏi.
- Nếu có nhiều cách hiểu khác nhau, hãy trình bày chúng - đừng âm thầm tự chọn một cách.
- Nếu có một cách tiếp cận đơn giản hơn, hãy nói ra. Phản biện lại khi có lý do chính đáng.
- Nếu có gì đó không rõ ràng, hãy dừng lại. Chỉ rõ điều gì gây khó hiểu. Hãy hỏi.

## 2. Ưu Tiên Sự Tối Giản

**Lượng code tối thiểu đủ để giải quyết vấn đề. Không viết code mang tính suy đoán/phòng hờ.**

- Không có tính năng nào ngoài những gì được yêu cầu.
- Không trừu tượng hóa (abstractions) cho các đoạn code chỉ dùng một lần.
- Không thêm sự "linh hoạt" hay "có thể cấu hình" nếu không được yêu cầu.
- Không xử lý lỗi cho các kịch bản bất khả thi.
- Nếu bạn viết 200 dòng mà thực tế chỉ cần 50 dòng, hãy viết lại.

Hãy tự hỏi: "Một kỹ sư Senior có cho rằng điều này là quá phức tạp không?" Nếu có, hãy đơn giản hóa nó.

## 3. Chỉnh Sửa Như Phẫu Thuật

**Chỉ chạm vào những gì bắt buộc phải chạm. Chỉ dọn dẹp đống lộn xộn do chính bạn tạo ra.**

Khi chỉnh sửa code có sẵn:
- Không "cải thiện" các đoạn code, comment hoặc định dạng xung quanh.
- Không tái cấu trúc (refactor) những thứ không bị hỏng.
- Phải khớp với văn phong (style) hiện có, ngay cả khi bạn có thói quen viết khác.
- Nếu bạn nhận thấy có dead code (code không bao giờ được chạy) không liên quan, hãy nhắc đến nó - đừng xóa nó.

Khi những thay đổi của bạn tạo ra các thành phần mồ côi:
- Hãy xóa các imports/biến/hàm mà do NHỮNG THAY ĐỔI CỦA BẠN khiến chúng không còn được sử dụng nữa.
- Không xóa các dead code đã tồn tại từ trước trừ khi được yêu cầu.

Bài kiểm tra: Mọi dòng code bị thay đổi đều phải liên kết trực tiếp đến yêu cầu của người dùng.

## 4. Hành Động Hướng Tới Mục Tiêu

**Xác định các tiêu chí thành công. Lặp lại cho đến khi được xác minh.**

Chuyển đổi các tác vụ thành các mục tiêu có thể xác minh được:
- "Thêm validation" → "Viết test cho các đầu vào không hợp lệ, sau đó làm cho chúng pass"
- "Sửa lỗi" → "Viết một test để tái tạo lại lỗi đó, sau đó làm cho nó pass"
- "Refactor X" → "Đảm bảo các test đều pass cả trước và sau khi làm"

Đối với các tác vụ gồm nhiều bước, hãy nêu ra một kế hoạch tóm tắt:
```
1. [Bước] → xác minh: [kiểm tra]
2. [Bước] → xác minh: [kiểm tra]
3. [Bước] → xác minh: [kiểm tra]
```

Tiêu chí thành công mạnh mẽ cho phép bạn lặp lại một cách độc lập. Tiêu chí yếu ("hãy làm cho nó chạy được") sẽ đòi hỏi phải làm rõ liên tục.

---

**Các nguyên tắc này đang phát huy hiệu quả nếu:** có ít thay đổi không cần thiết hơn trong các bản diff, ít phải viết lại code hơn do làm quá phức tạp vấn đề, và các câu hỏi làm rõ được đặt ra trước khi tiến hành code chứ không phải sau khi đã mắc lỗi.

## 5. Quy trình thao tác

**Được sử dụng với mỗi lần chạy 1 session chat để đảm bảo bộ nhớ, tiến độ dự án**

- Luôn đọc folder memory dự án để nắm tiến độ, những gì đã được giải quyết, những gì còn tồn đọng. Từ đó tránh việc lặp lại công việc, hoặc đưa ra các giải pháp mà trước đó đã bị loại bỏ.
- Luôn ghi lại và cập nhật progress vào memory.
- Lưu ý file quy định chung, bắt buộc đọc, tuyệt đối không được sửa, xoá nếu chưa được sự cho phép, vì file này quy định những điều cơ bản để tránh xoá, thao tác sai ảnh hưởng đến bên ngoài. 
Ví dụ: khi tiến hành lên server production, truy cập vào database production,.... File chung này cũng sẽ là nơi lưu những hướng dẫn như cách ssh, server đang ở đâu, domain như nào, file .env có những gì, database đang dùng là gì, truy cập bằng gì,...

- 1 file md memory đảm bảo các yếu tố sau:
    + **Time**: ghi rõ ngày giờ thực hiện, bắt đầu chat, chat lần cuối
    + **Progress**: những gì đã được giải quyết, những gì còn tồn đọng
    + **Todo**: những gì cần làm tiếp
    + **Note**: những gì cần lưu ý, ghi nhớ
    + **Nội dung**: những thứ vắn tắt để có thể nắm tình hình:
        - List task đã được lên kế hoạch (kể cả những task đã xong) để nắm tiến độ, tránh lặp lại công việc, hoặc đưa ra các giải pháp mà trước đó đã bị loại bỏ.
        - Những thay đổi gần đây, lỗi phát sinh trong quá trình phát triển task, tiến trình session chính đó.
        - Những giải pháp đã được xem xét (kể cả những giải pháp đã bị loại bỏ) để tránh lặp lại công việc, hoặc đưa ra các giải pháp mà trước đó đã bị loại bỏ.
        - Những điều cần lưu ý, những điều cần phải để tâm đến trong quá trình phát triển, hoặc những thông tin vắn tắt để có thể nắm tình hình.
```



**Lợi ích của quy trình này là** giảm thiểu các lỗi do thiếu thông tin, tránh lặp lại công việc, đảm bảo tiến độ dự án, và tăng hiệu quả làm việc của AI Agent.