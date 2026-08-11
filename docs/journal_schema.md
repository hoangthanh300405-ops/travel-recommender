# journal_log.csv — Schema

Nguồn rating cho user thật, thay thế/bổ sung cho UCI dataset. Tương thích trực tiếp
với `load_journal()` trong `03_journal_to_wide.py` (bắt buộc: `user_id`, `google_category`, `rating`).

| Cột | Bắt buộc | Kiểu | Ghi chú |
|---|---|---|---|
| `entry_id` | không | int | id duy nhất mỗi lần ghi, tiện sửa/xoá về sau |
| `user_id` | **có** | string | khớp với id dùng trong `df_wide_journal` / `cf_model` |
| `date` | không (nên có) | YYYY-MM-DD | dùng cho next-destination recommendation (thứ tự thời gian) và itinerary sau này |
| `place_name` | không (nên có) | string | tên địa điểm thật — cần cho việc build custom Vietnamese POI dataset qua Overpass sau này |
| `google_category` | **có** | string | text tự do, sẽ được map qua `category_crosswalk.csv` (02_category_crosswalk.py) sang 1 trong 24 category UCI |
| `province_city` | không | string | scope địa lý — hữu ích khi lọc theo vùng cho itinerary |
| `latitude` / `longitude` | không (bắt buộc cho itinerary sau) | float | cần cho OR-Tools TSP; nếu thiếu có thể geocode bổ sung sau bằng Overpass/Nominatim |
| `rating` | **có** | float 1–5 | đúng thang UCI; `load_journal()` tự loại các dòng ngoài [1,5] |
| `note` | không | string | ghi chú tự do — không dùng cho scoring hiện tại, để dành cho phân tích/NLP sau này nếu cần |

## Vì sao chọn các cột này

- `google_category` (chứ không phải category UCI trực tiếp) — giữ nguyên đúng vai trò
  cầu nối mà `category_crosswalk.csv` đã thiết kế: user ghi tên category tự nhiên,
  hệ thống tự map sang 24 category chuẩn để tái dùng `cf_model`/`item_sim` đã train.
- `latitude`/`longitude` để trống được ở giai đoạn hiện tại (chỉ cần cho phase 3 —
  itinerary), nhưng đưa vào schema từ đầu để không phải sửa lại format khi tới lúc cần.
- `date` không bắt buộc cho hybrid rating hiện tại, nhưng **bắt buộc về mặt thiết kế**
  cho next-destination recommendation (cần biết thứ tự user đã đi những đâu).

## Việc cần làm tiếp

1. Xác nhận `user_id` sẽ lấy từ đâu (tài khoản thật, hay id tạm thời cho demo).
2. Quyết định `google_category` nhập tự do hay chọn từ danh sách gợi ý (ảnh hưởng
   tỷ lệ "unmapped" khi qua crosswalk).
3. Nếu muốn itinerary sớm, nên bắt buộc `latitude`/`longitude` ngay từ đầu thay vì
   để optional.
