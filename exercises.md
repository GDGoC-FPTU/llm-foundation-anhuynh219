# Ngày 1 — Bài Tập & Phản Ánh
## Nền Tảng LLM API | Phiếu Thực Hành

**Thời lượng:** 1:30 giờ  
**Cấu trúc:** Lập trình cốt lõi (60 phút) → Bài tập mở rộng (30 phút)

---

## Phần 1 — Lập Trình Cốt Lõi (0:00–1:00)

Chạy các ví dụ trong Google Colab tại: https://colab.research.google.com/drive/172zCiXpLr1FEXMRCAbmZoqTrKiSkUERm?usp=sharing

Triển khai tất cả TODO trong `template.py`. Chạy `pytest tests/` để kiểm tra tiến độ.

**Điểm kiểm tra:** Sau khi hoàn thành 4 nhiệm vụ, chạy:
```bash
python template.py
```
Bạn sẽ thấy output so sánh phản hồi của GPT-4o và GPT-4o-mini.

---

## Phần 2 — Bài Tập Mở Rộng (1:00–1:30)

### Bài tập 2.1 — Độ Nhạy Của Temperature
Gọi `call_openai` với các giá trị temperature 0.0, 0.5, 1.0 và 1.5 sử dụng prompt **"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

(Vì chỉ có Gemini API key nên thí nghiệm này được chạy bằng `call_gemini` với `gemini-2.5-flash`, kết quả lưu ở `temp_results.txt`.)

| Temp | Chủ đề chính | Độ dài (output tokens) |
| --- | --- | --- |
| 0.0 | Việt Nam xuất khẩu hạt điều lớn nhất thế giới (formal, có giải thích) | 109 |
| 0.5 | "Phở không phải là món ăn sáng truyền thống" — kèm danh sách 4 món sáng khác | 236 |
| 1.0 | Lại quay về chủ đề hạt điều, nhưng ngắn gọn hơn nhiều | 63 |
| 1.5 | Nón lá đa năng — chống nắng, chống mưa, làm quạt, làm giỏ, thậm chí "múc nước" | 229 |

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Temperature càng thấp (0.0) thì model chọn fact "an toàn" và phổ biến nhất (hạt điều) với cách trình bày khá khuôn mẫu. Khi tăng dần lên 0.5 và 1.5, chủ đề trở nên đa dạng và sáng tạo hơn — xuất hiện những góc nhìn bất ngờ như "Phở không phải món sáng truyền thống" hay chi tiết "dùng nón lá để múc nước". Tuy nhiên temperature cao cũng làm độ dài và mức độ tập trung dao động mạnh giữa các lần gọi, không còn dự đoán được.

**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ đặt **temperature ≈ 0.2** (gần như deterministic nhưng vẫn cho phép thay đổi nhỏ về văn phong). Chatbot CSKH cần câu trả lời nhất quán, đúng chính sách và có thể audit được — hai khách hỏi cùng câu hỏi không nên nhận hai câu trả lời mâu thuẫn. Temperature thấp cũng giảm rủi ro hallucinate thông tin sai về giá, chính sách hoàn tiền, hay quy trình.

---

### Bài tập 2.2 — Đánh Đổi Chi Phí
Xem xét kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người thực hiện 3 lần gọi API, mỗi lần trung bình ~350 token.

**Ước tính xem GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này:**
> Tổng tokens/ngày = 10.000 × 3 × 350 = **10,5 triệu tokens**. Giả định tỉ lệ input/output là 50/50 → 5,25M input + 5,25M output.
> - **GPT-4o**: 5,25 × $5 + 5,25 × $20 = $26,25 + $105 = **$131,25/ngày** (≈ $3.937/tháng).
> - **GPT-4o-mini**: 5,25 × $0,15 + 5,25 × $0,60 = $0,79 + $3,15 = **$3,94/ngày** (≈ $118/tháng).
> - **Tỉ lệ**: $131,25 / $3,94 ≈ **33,3 lần**. GPT-4o đắt gấp ~33× cho cùng khối lượng tokens.

**Mô tả một trường hợp mà chi phí cao hơn của GPT-4o là xứng đáng, và một trường hợp GPT-4o-mini là lựa chọn tốt hơn:**
> **Nên dùng GPT-4o**: trợ lý lập trình hoặc phân tích pháp lý/y tế — nơi một lỗi reasoning có thể tốn nhiều giờ debug của kỹ sư (giá hàng trăm USD/giờ) hoặc gây hậu quả pháp lý. Chênh lệch $0,01–0,02 mỗi truy vấn không đáng kể so với chi phí sai lệch.
> **Nên dùng GPT-4o-mini**: phân loại sentiment, tóm tắt email, gắn tag sản phẩm, moderation cấp 1 — tác vụ đơn giản, mẫu rõ ràng, có thể chấp nhận sai sót nhỏ và có pipeline human-review phía sau. Tiết kiệm 33× chi phí mỗi ngày là khoản tiền thật.

---

### Bài tập 2.3 — Trải Nghiệm Người Dùng với Streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất ở các giao diện hội thoại realtime mà người dùng đang **chờ và đọc** câu trả lời — chatbot, copilot lập trình, trợ lý viết — vì độ trễ tới token đầu tiên thường dưới 1 giây trong khi tổng câu trả lời có thể mất 10–30 giây; thấy chữ chạy ra giúp người dùng cảm nhận hệ thống "đang sống" và không bỏ giữa chừng. Ngược lại, non-streaming hợp hơn khi đầu ra phải được **xử lý nguyên khối** trước khi dùng được: gọi API trả về JSON cần parse và validate, pipeline batch chạy nền (tóm tắt 10.000 email overnight), hay các bước trung gian trong một agent pipeline mà bước sau cần nguyên kết quả bước trước — ở những kịch bản này streaming không cải thiện UX mà chỉ làm code phức tạp thêm.


## Danh Sách Kiểm Tra Nộp Bài
- [x] Tất cả tests pass: `pytest tests/ -v`
- [x] `call_openai` đã triển khai và kiểm thử
- [x] `call_openai_mini` đã triển khai và kiểm thử
- [x] `compare_models` đã triển khai và kiểm thử
- [x] `streaming_chatbot` đã triển khai và kiểm thử
- [x] `retry_with_backoff` đã triển khai và kiểm thử
- [x] `batch_compare` đã triển khai và kiểm thử
- [x] `format_comparison_table` đã triển khai và kiểm thử
- [x] `exercises.md` đã điền đầy đủ
- [x] Sao chép bài làm vào folder `solution` và đặt tên theo quy định
