###### TRƯỜNG ĐẠI HỌC CÔNG NGHỆ - ĐẠI HỌC QUỐC GIA HÀ NỘI

### Một mô hình đa phương thức cho phát hiện căng thẳng

### trong hoạt động hàng ngày

```
Người thực hiện : Nguyễn Công Minh - 22028148
Giảng viên hướng dẫn : TS. Vũ Thị Hồng Nhạn
```

## Nội dung

#### 01

```
VẤN ĐỀ VÀ
MỤC TIÊU
NGHIÊN CỨU
```
#### 02

```
THIẾT KẾ
HỆ THỐNG VÀ
PHƯƠNG PHÁP ĐỀ XUẤT
```
#### 03

```
THỰC NGHIỆM
VÀ KẾT QUẢ
```
#### 04

```
HẠN CHẾ VÀ
HƯỚNG PHÁT
TRIỂN
```
#### 05

**KẾT LUẬN**


##### 01. Đặt vấn đề

```
WHO: đại dịch COVID-19 tăng 25% rối loạn lo âu và trầm
cảm toàn cầu.
Stress kéo dài gây bệnh tim mạch, suy giảm miễn dịch,
trầm cảm.
Phương pháp đánh giá hiện tại: bảng hỏi hồi cứu — chủ
quan, không liên tục, gây gián đoạn.
```
**Mục tiêu:** Dự đoán stress liên tục (1–9) bằng pipeline context-aware có thể giải thích được


**Xu hướng phổ biến Hướng tiếp cận của khóa luận**

Phân loại stress rời rạc Hồ i quy stress liên tục (1–9)

Nhận diện hành động và stress xử lý tách rời với Context-Stress ModifierPipeline thống nhất^

```
So sánh mô hình chưa
công bằng
```
```
Benchmark 5 kiến trúc trên cùng
pipeline
```
Giải thích kết quả hạn chế 4 phtrưng + phân tích lôương pháp phẫn tích đi đa chiêặ̀uc

##### 01. Đặt vấn đề

Các khoảng trống nghiên cứu


##### 02. Thiết kế hệ thống

Kiến trúc tổng thể triển khai

```
1.HAR Module: sử dụng dữ liệu WISDM và mô hình Stacked Bi-LSTM để nhận dạng 6 hoạt động
→ 96.19% accuracy.
```
```
2.Data Generation: sinh tập dữ liệu bán mô phỏng đa phương thức có
Context-Stress Modifier.
```
```
3.Feature & Pipeline: Split → Encode → Normalize → Sequence (chống
leakage).
```
```
4.Model & Evaluation: huấn luyện, Bayesian tuning, so sánh 5 kiến trúc
và giải thích kết quả.
```

```
Lưu ý: Đây là kiến trúc triển khai tổng thể. Không phải toàn bộ 4 module đều là
đóng góp mới; một số thành phần là kế thừa/ứng dụng lại để xây dựng pipeline.
```

##### 02. Phương pháp đề xuất

**Phần đề xuất thực sự của khóa luận**

```
1. Ghép nhãn hoạt động từ HAR vào bài toán stress:
   Activity không chỉ là nhãn phụ, mà trở thành ngữ cảnh để diễn giải tín hiệu sinh lý.
```

```
2. Bộ sinh dữ liệu đa phương thức có truy vết:
   Kết hợp accelerometer thực từ WISDM với các biến mô phỏng như Heart_Rate,
   Screen_Usage, Mood, Energy, Sleep, Location và Stress_Level.
```

```
3. Context-Stress Modifier:
   Điều chỉnh stress theo tổ hợp Activity × Location × Context để xử lý nhập nhằng sinh lý,
   ví dụ HR cao do Jogging khác HR cao khi Sitting + Work + deadline.
```

```
4. Protocol kiểm chứng:
   Split theo thời gian → Encode/Normalize chỉ fit trên train → Sequence → Benchmark 5 mô hình
   và giải thích bằng feature importance.
```

**Kế thừa / sử dụng lại:** WISDM/Kwapisz et al. (2011), Bi-LSTM/LSTM, Bayesian Optimization,
MET Compendium/Ainsworth, các mô hình stress trong y văn.

**Đề xuất mới trong đồ án:** cách kết hợp các nguồn đó thành một pipeline context-aware có thể
truy vết công thức, kiểm chứng giả thuyết và phân tích được.

##### 03. Thiết kế

Module 1: Nhận diện hành động

```
Điểm then chốt: Nhãn hoạt động từ Module 1 là đầu vào bắt buộc cho Module 2 — tạo nên tính
nhận thức ngữ cảnh của toàn hệ thống
```
```
Mô hình: Stacked Bi-LSTM (2 lớp)
Accuracy : 96.19%
```
Tập data gốc gồm 1,098,207 bản ghi của Wireless Sensor Data Mining (WISDM) Lab:

```
Timestamp X-acceleration Y-acceleration Z-acceleration Activity
49105962326000 - 6. 946. 377 12. 680544 0. 50395286 Jogging
11281572276000 - 3. 49 17. 77 1. 5390993 Walking
4133402202000 3. 38 6. 28 - 2. 1111538 Upstairs
12428452283000 3. 3 9. 3 1. 0760075 Sitting
14593872297000 - 1. 27 4. 02 1. 2666923 Downstairs
14953002210000 - 0. 11 9. 58 2. 4925237 Standing
```


```
Nhóm Số lượng Đặc trưng
Thời gian 2 Hour, Day_of_Week
```
```
Hoạt động và cảm biến 4 ActivAitcyc,^ eAlcecroemleerotemre_Yte,r_X,^
Accelerometer_Z
Sinh lý 1 Heart_Rate
```
Hành vi (^3) PShcorneee_nE_vUesnatg_eF_rCequurreenncty,,
Mood_Score
Ngữ cảnh 3 LocaStiloenep,^ E_Dneurrgayti_oLnevel,^

##### 03. Thiết kế

Module 2: Tạo dữ liệu đa phương thức

**Chọn 13 đặc trưng cho tập dữ liệu**

```
Tại sao dùng dữ liệu bán mô phỏng?
Hạn chế về thiết bị và kỹ thu thập dữ liệu thô
Kiểm soát biến thực nghiệm có hệ thống
Kiểm chứng giả thuyết về Context-Stress
Modifier
Dữ liệu gia tốc kế thực từ WISDM + hành vi mô
phỏng có cơ sở khoa học
```
```
Phạm vi: Nhãn stress và hệ số hành vi là heuristic có neo nghiên cứu, phục
vụ kiểm chứng phương pháp, không thay thế nhãn lâm sàng/EMA.
```

```
Khung giờ Hoạt động chính Vị trí Ý nghĩa với stress
06 : 00 – 08 : 00 Thức dậy, sinh hoạt cá nhân Home Trạng thái hồi phục
08 : 00 – 09 : 00 Di chuyển đi làm Commute Tăng nhẹ do giao thông
09 : 00 – 12 : 00 Làm việc buổi sáng Work Áp lực tăng dần
12 : 00 – 13 : 00 Nghỉ trưa Work/Outdoor Giảm tạm thời
13 : 00 – 17 : 00 Làm việc buổi chiều Work Áp lực cao nhất ngày
17 : 00 – 18 : 00 Di chuyển về nhà Commute Chuyển pha stress
18 : 00 – 20 : 00 Ăn tối/tập luyện Home/Gym Phục hồi dần
20 : 00 – 22 : 30 Giải trí, nghỉ ngơi Home Giảm stress rõ rệt
22 : 30 – 06 : 00 Ngủ Home Reset sinh lý ngày hôm sau
```
##### 03. Thiết kế

Module 2: Tạo dữ liệu đa phương thức

**Chi tiết lịch trình mô phỏng trong 30 ngày**

```
•Vị trí: home, commute, work, gym,
social, outdoor.
```
- Kết hợp sự kiện đặc biệt
(6%/ngày: deadline, ốm, thi
cử) + chu kỳ tuần/tháng.
- 2 mẫu/phút trong khung thức tạo
khoảng 1,815 mẫu/ngày → 54,
mẫu trong 30 ngày.


```
Theo nghiên cứu của McEwen: allostatic load, stress/recovery,
cơ thể tích lũy gánh nặng khi thiếu phục hồi.
```
##### 03. Thiết kế

Module 2: Tạo dữ liệu đa phương thức

**Mô phỏng mức năng lượng**

```
Bao gồm dao động theo chu kỳ tháng, mệt mỏi tích lũy theo tuần và nhiễu sinh
hoạt hằng ngày
```

**HRrest** : lấy từ hồ sơ người dùng

**∆HRactivity** : quy đổi thực dụng từ MET (Ainsworth, 2011)

```
∆HRstress = (StressLevel − 4) × 3 bpm — heuristic từ cơ chế
hệ giao cảm.
```
**∆HRfatigue** = (1 − E) × 5 bpm — bù trừ khi kiệt sức

##### 03. Thiết kế

Module 2: Tạo dữ liệu đa phương thức

**Mô hình hóa nhịp tim theo bối cảnh**


**HRrest** : lấy từ hồ sơ người dùng

**∆HRactivity** : quy đổi thực dụng từ MET (Ainsworth, 2011)

```
∆HRstress = (StressLevel − 4) × 3 bpm — heuristic từ cơ chế
hệ giao cảm.
```
**∆HRfatigue** = (1 − E) × 5 bpm — bù trừ khi kiệt sức

##### 03. Thiết kế

Module 2: Tạo dữ liệu đa phương thức

**Mô hình hóa nhịp tim theo bối cảnh**


##### 03. Thiết kế

Module 2: Tạo dữ liệu đa phương thức

**Bảng điều chỉnh nhịp tim theo hoạt động dựa trên giá trị MET**

**Theo American Heart Association, nhịp tim trung bình của người trưởng thành khỏe mạnh:**

**Tanaka et al. (2001):**


**HRrest** : lấy từ hồ sơ người dùng

**∆HRactivity** : quy đổi thực dụng từ MET (Ainsworth, 2011)

```
∆HRstress = (StressLevel − 4) × 3 bpm — heuristic từ cơ chế
hệ giao cảm.
```
**∆HRfatigue** = (1 − E) × 5 bpm — bù trừ khi kiệt sức

##### 03. Thiết kế

Module 2: Tạo dữ liệu đa phương thức

**Mô hình hóa nhịp tim theo bối cảnh**


**Hoạt động MET ∆HRactivity (bpm) Cơ sở**

```
Sitting 1. 0 – 1. 5 + 0 Nghỉ ngơi, không vận động
```
```
Standing 1. 5 – 2. 0 + 6 Đứng thụ động, tăng nhẹ
```
```
Walking 3. 0 – 3. 5 + 18 Đi bộ bình thường ( 4 – 5 km/h)
```
```
Downstairs 35 + 22 Tương đương đi bộ nhanh
```
```
Upstairs 5. 0 – 8. 0 + 28 Leo cầu thang, cường độ cao
```
```
Jogging 7. 0 – 8. 0 + 40 Chạy bộ vừa phải ( 8 km/h)
```
##### 03. Thiết kế

Module 2: Tạo dữ liệu đa phương thức

**Bảng điều chỉnh nhịp tim theo hoạt động dựa trên giá trị MET**


**HRrest** : lấy từ hồ sơ người dùng

**∆HRactivity** : quy đổi thực dụng từ MET (Ainsworth, 2011)

```
∆HRstress = (StressLevel − 4) × 3 bpm — heuristic từ cơ chế
hệ giao cảm.
```
**∆HRfatigue** = (1 − E) × 5 bpm — bù trừ khi kiệt sức

##### 03. Thiết kế

Module 2: Tạo dữ liệu đa phương thức

**Mô hình hóa nhịp tim theo bối cảnh**


**HRrest** : lấy từ hồ sơ người dùng

**∆HRactivity** : quy đổi thực dụng từ MET (Ainsworth, 2011)

```
∆HRstress = (StressLevel − 4) × 3 bpm — heuristic từ cơ chế
hệ giao cảm.
```
**∆HRfatigue** = (1 − E) × 5 bpm — bù trừ khi kiệt sức

##### 03. Thiết kế

Module 2: Tạo dữ liệu đa phương thức

**Mô hình hóa nhịp tim theo bối cảnh**

```
Ý nghĩa: Nhịp tim không phải giá trị tĩnh — mà là tổng
hợp từ hoạt động, stress, mệt mỏi và nhiễu sinh lý
```

##### 03. Thiết kế

Module 2: Tạo dữ liệu đa phương thức

**Công thức tính mức độ stress đa yếu tố**

**Tính quán tính stress:**

- ∆S_sleep: ∆S_sleep = (1 − Q_sleep) × 2 (McEwen, 2008)
- ∆S_momentum = (S_recent − 4) × 0.

**Cơ sở lý thuyết:**

- ∆S_time: nhịp cortisol (Chrousos, 2009)
- ∆S_activity: endorphin khi vận động (Salmon, 2001)
- ∆S_location: mô hình Demand-Control (Karasek, 1979)
- ∆S_HR: (Hovsepian,2015)


##### 03. Thiết kế

Module 2: Tạo dữ liệu đa phương thức

**Công thức tính mức độ stress đa yếu tố**

**Tính quán tính stress:**

- ∆S_sleep: ∆S_sleep = (1 − Q_sleep) × 2 (McEwen, 2008)
- ∆S_momentum = (S_recent − 4) × 0.

**Cơ sở lý thuyết:**

- ∆S_time: nhịp cortisol (Chrousos, 2009)
- ∆S_activity: endorphin khi vận động (Salmon, 2001)
- ∆S_location: mô hình Demand-Control (Karasek, 1979)
- ∆S_HR: (Hovsepian,2015)


```
Khung giờ ∆Stime Cơ sở sinh lý
```
```
7 h– 9 h + 0. 5 Hiệ 3 n 0 t–ượ 45 n gp^ hcúortt đisâòul stăanug k^5 h^0 i t–h^7 ứ^5 c% d^ tậryong^
```
```
9 h– 12 h + 1. 0 Cortáispo llự^ đcỉ nchô^ nbgu ổviiệ^ scáng,^
```
```
13 h– 17 h + 1. 5 Comrệtits molỏ^ cia bou^ +ổ^ ti ícchhi^ êl̀ũuy^
```
```
18 h– 20 h − 0. 5 Corntigsơoil bguiảổmi t,^ ôńighỉ^
```
```
Sau 20 h − 0. 8 Cortisol thấp nhất, chuẩn bị ngủ
```
##### 03. Thiết kế

Module 2: Tạo dữ liệu đa phương thức

**Điều chỉnh theo thời gian
Cơ sở:** Mô phỏng nhịp sinh học cortisol hàng ngày theo Chrousos:


##### 03. Thiết kế

Module 2: Tạo dữ liệu đa phương thức

**Công thức tính mức độ stress đa yếu tố**

**Tính quán tính stress:**

- ∆S_sleep: ∆S_sleep = (1 − Q_sleep) × 2 (McEwen, 2008)
- ∆S_momentum = (S_recent − 4) × 0.

**Cơ sở lý thuyết:**

- ∆S_time: nhịp cortisol (Chrousos, 2009)
- ∆S_activity: endorphin khi vận động (Salmon, 2001)
- ∆S_location: mô hình Demand-Control (Karasek, 1979)
- ∆S_HR: (Hovsepian,2015)


```
Hoạt động ∆Sactivity Cơ sở khoa học
```
```
Jogging − 0. 8 Giải^ phóngc^ oerntdisoorlphin,^ giảm
```
```
Standing + 0. 1 Gần như trung tính
```
```
Sitting + 0. 2 Ít vận động, tăng nhẹ căng thẳng
```
##### 03. Thiết kế

Module 2: Tạo dữ liệu đa phương thức

**Điều chỉnh theo hoạt động
Cơ sở:** Nghiên cứu của Salmon et al. (2001) cho rằng luyện tập thể chất có thể giảm lo âu, trầm cảm và độ nhạy với
stress:


##### 03. Thiết kế

Module 2: Tạo dữ liệu đa phương thức

**Công thức tính mức độ stress đa yếu tố**

**Cơ sở lý thuyết:**

- ∆S_time: nhịp cortisol (Chrousos, 2009)
- ∆S_activity: endorphin khi vận động (Salmon, 2001)
- ∆S_location: mô hình Demand-Control (Karasek, 1979)
- ∆S_HR: (Hovsepian,2015)

**Tính quán tính stress:**

- ∆S_sleep: ∆S_sleep = (1 − Q_sleep) × 2 (McEwen, 2008)
- ∆S_momentum = (S_recent − 4) × 0.3


```
Vị trí ∆Slocation Cơ sở
Work + 1. 5 Áp^ lựDce^ mcôanngd^ v-Ciệocn^ (tmroôl^ )hình
Home − 0. 5 Mtôhiu^ tộrườc, anng tqouàenn^
Commute + 1. 0 Đi lại căng thẳng, giao thông
Gym − 0. 3 Môi^ trườngs^ tvrậens^ sđộng,^ giảm
Social − 0. 4 Hỗ trợ xã hội giảm stress
```
##### 03. Thiết kế

Module 2: Tạo dữ liệu đa phương thức

**Điều chỉnh theo vị trí
Cơ sở:** Nghiên cứu của Bratman et al. (2015) về tác động tích cực của môi trường tự nhiên đối với giảm stress:


##### 03. Thiết kế

Module 2: Tạo dữ liệu đa phương thức

**Công thức tính mức độ stress đa yếu tố**

**Cơ sở lý thuyết:**

- ∆S_time: nhịp cortisol (Chrousos, 2009)
- ∆S_activity: endorphin khi vận động (Salmon, 2001)
- ∆S_location: mô hình Demand-Control (Karasek, 1979)
- ∆S_HR: (Hovsepian,2015)

**Tính quán tính stress:**

- ∆S_sleep: ∆S_sleep = (1 − Q_sleep) × 2 (McEwen, 2008)
- ∆S_momentum = (S_recent − 4) × 0.3


##### 03. Thiết kế

Module 2: Tạo dữ liệu đa phương thức

**Điều chỉnh theo nhịp tim
Cơ sở:** Theo Hovsepian (2015) nhịp tim tăng khi nghỉ là dấu hiệu kích hoạt hệ giao cảm, tương quan dương với stress:


##### 03. Thiết kế

Module 2: Tạo dữ liệu đa phương thức

**Công thức tính mức độ stress đa yếu tố**

**Cơ sở lý thuyết:**

- ∆S_time: nhịp cortisol (Chrousos, 2009)
- ∆S_activity: endorphin khi vận động (Salmon, 2001)
- ∆S_location: mô hình Demand-Control (Karasek, 1979)
- ∆S_HR: (Hovsepian,2015)

**Tính quán tính stress:**

- ∆S_sleep: McEwen, 2008
- ∆S_momentum = (S_recent − 4) × 0.3


##### 03. Thiết kế

Module 2: Tạo dữ liệu đa phương thức

**Điều chỉnh theo giấc ngủ
Cơ sở:** Theo McEwen (2008), thiếu ngủ làm tăng cortisol ngày hôm sau và giảm khả năng điều hòa cảm xúc:

với Qsleep ∈ [0, 1] là chất lượng giấc ngủ.


##### 03. Thiết kế

Module 2: Tạo dữ liệu đa phương thức

**Công thức tính mức độ stress đa yếu tố**

**Cơ sở lý thuyết:**

- ∆S_time: nhịp cortisol (Chrousos, 2009)
- ∆S_activity: endorphin khi vận động (Salmon, 2001)
- ∆S_location: mô hình Demand-Control (Karasek, 1979)
- ∆S_HR: (Hovsepian,2015)

**Tính quán tính stress:**

- ∆S_sleep: McEwen, 2008
- ∆S_momentum = (S_recent − 4) × 0.3


##### 03. Thiết kế

Module 2: Tạo dữ liệu đa phương thức

**Hiệu ứng quán tính
Cơ sở:** Theo Plarre (2007) stress có xu hướng duy trì theo thời gian:

S_recent là trung bình stress của 3 mẫu gần nhất


Bact(at) là cường độ sử dụng điện thoại nền theo hoạt động

Mloc(lt) là hệ số theo vị trí

##### 03. Thiết kế

Module 2: Tạo dữ liệu đa phương thức

**Mô phỏng mức sử dụng màn hình hiện tại**

```
Nghiên cứu của Lane et al. 2010 : Activity và Location quyết định
khả năng dùng điện thoại.
```
```
Stress chỉ làm thay đổi nhẹ, không chi
phối toàn bộ feature.
```

```
Hoạt động Bact Vị trí Mloc
Sitting 0. 7 home 1. 3
Standing 0. 4 work 1. 1
Walking 0. 2 commute 0. 9
Jogging 0. 05 outdoor 0. 6
Upstairs 0. 1 social 0. 4
Downstairs 0. 15 gym 0. 3
```
##### 03. Thiết kế

Module 2: Tạo dữ liệu đa phương thức

**Mô phỏng mức sử dụng màn hình hiện tại**

**Cơ sở:** Nghiên cứu của Lane et al. 2010: Activity và Location quyết định khả năng dùng điện thoại.


```
Stress cao kéo mood xuống vừa phải,
chứ không hoàn toàn quyết định
```
##### 03. Thiết kế

Module 2: Tạo dữ liệu đa phương thức

**Mô phỏng điểm tâm trạng**

**Cơ sở:** Nghiên cứu smartphone sensing như MoodScope, StudentLife cho thấy mood liên quan đến hành vi,
sleep, activity, workload

```
Mood nền của ngày, được sinh từ
nhiễu hàng ngày và sự kiện đặc biệt.
```

Hoạt động Ngữ cảnh δ(a, l, c) Giải thích

Sitting

```
Work, sáng, workload cao + 1. 5 Áp lực công việc sáng
Home, tối, workload thấp − 1. 5 Nghỉ ngơi tại nhà
```
Walking

```
Work, chiều, workload cao + 1. 6 Đi họp gấp
Commute, sáng, ngày thường + 0. 8 Giờ cao điểm
Outdoor, tối, workload thấp − 1. 2 Đi dạo thư giãn
```
Jogging

```
Outdoor, sáng, cuối tuần − 1. 5 Chạy bộ sáng cuối tuần
Gym, tối, ngày thường − 0. 8 Tập sau giờ làm
```
##### 03. Thiết kế

Module 2: Tạo dữ liệu đa phương thức

**Cơ chế Context-Stress Modifier**

**Cơ sở lý thuyết:** (Lazarus & Folkman, 1984): Cùng một tín hiệu sinh lý, ý nghĩa stress hoàn toàn khác nhau tùy ngữ cảnh.


##### 03. Thiết kế

Module 2: Tạo dữ liệu đa phương thức

**Cơ chế Context-Stress Modifier**

**Cơ sở lý thuyết:** theo nghiên cứu của McEwen (2008) và Yoo (2007 ), thiếu ngủ làm tăng phản ứng với stress


##### 03. Thiết kế

Module 2: Tạo dữ liệu đa phương thức

**Cơ chế Context-Stress Modifier**

**Cơ sở lý thuyết:** theo nghiên cứu của Bratman (2015), môi trường tự nhiên, yên tĩnh giúp hỗ trợ hiệu ứng phục hồi tinh thần


##### 03. Thiết kế

Module 2: Tạo dữ liệu đa phương thức

**Cơ chế Context-Stress Modifier**

**Cơ sở lý thuyết:** theo nghiên cứu của Lazarus & Folkman (1984) + Wang (Student Life 2014), đánh giá nhận thức về tình
huống xã hội ảnh hưởng đến stress


##### 03. Thiết kế

Module 2: Tạo dữ liệu đa phương thức

**Cơ chế Context-Stress Modifier**


**Sequential Split**

**Label Encoding**

**Standard Normalization**

**Sequence**

```
Dữ liệu 54.448 mẫu được chia theo thứ tự thời
gian (không shuffle) với tỷ lệ 70/15/15:
Train = data[0 : ntrain] với ntrain = ⌊0.70 × N ⌋
= 38,113
Val = data[ntrain : nval] với nval = ⌊0.85 × N ⌋
= 46,280
Test = data[nval : N ] = 8,168 mẫu
```
##### 03 Thiết kế

Module 3: Tiền xử lý dữ liệu


**Sequential Split**

**Label Encoding**

**Standard Normalization**

**Sequence**

```
Hai biến phân loại được mã hóa thành số nguyên
bằng LabelEncoder của scikit-learn.
Encoder được fit chỉ trên tập train, sau đó
transform cho val và test → Chống data leakage
```
```
Biến Số lớp Giá trị
Activity 6 Downstairs, Jogging, Sitting, Standing, Upstairs, Walking
Location 6 commute, gym, home,outdoor, social, wor^
```
##### 03 Thiết kế

Module 3: Tiền xử lý dữ liệu


**Sequential Split**

**Label Encoding**

**Standard Normalization**

**Sequence**

##### 03 Thiết kế

Module 3: Tiền xử lý dữ liệu

```
μtrain và σtrain là giá trị trung bình và độ lệch
chuẩn của feature trên tập train.
Các tập val và test dùng cùng μtrain, σtrain —
không fit lại.
```

**Sequential Split**

**Label Encoding**

**Standard Normalization**

**Sequence**

##### 03 Thiết kế

Module 3: Tiền xử lý dữ liệu

```
Mỗi bản ghi tại một thời điểm được biểu diễn thành
vector 13 chiều
Dữ liệu được biến đổi thành chuỗi bằng cửa sổ
trượt (sliding window) với độ dài
seq_length = 60:
```
```
Mỗi cửa sổ gồm 60 vector liên tiếp, mỗi vector có 13
feature
```

```
Lớp Tên lớp Output Shape Params
1 Input ( 60 , 13 ) 0
2 Bidirectional(LSTM( 128 , return_seq=True)) ( 60 , 256 ) 45. 408
3 Dropout( 0. 3 ) ( 60 , 256 ) 0
4 Bidirectional(LSTM( 64 )) ( 128 ) 164. 096
5 Dropout( 0. 3 ) ( 128 ) 0
6 Dense( 64 , ReLU) ( 64 ) 8. 256
7 Dropout( 0. 3 ) ( 64 ) 0
8 Dense( 32 , ReLU) ( 32 ) 2. 080
9 Dense( 1 , Linear) ( 1 ) 33
Tổng tham số 320. 129
```
##### 03 Thiết kế

Module 4: Kiến trúc Baseline Stacked Bi-LSTM

```
Dropout = 0.3 quá lớn →
underfitting
Learning rate = 0.001 →
hội tụ chậm
320K params →
over-parameterized
```
```
→ Cần cải thiện bằng tối
ưu tham số
```

```
Tham số Baseline Tuned Nhận xét
lstm_units_ 1 128 64 Giảm 50 % — mô hình nhỏ hơn
lstm_units_ 2 64 64 Giữ nguyên
dropout_rate 0. 3 0. 1 Giảm mạnh — ít regularization hơn
dense_units 64 128 Tăng — mở rộng lớp Dense
learning_rate 0. 001 0. 01 Tăng 10 × — hội tụ nhanh hơn
Tổng params 320. 129 163. 585 Giảm 48. 9 %
```
- Dropout: 0.3 → 0.1 — giải
phóng năng lực biểu diễn
- Learning rate: 0.001 → 0.01 —
thoát local minima nhanh hơn
- LSTM units lớp 1: 128 → 64 —
mô hình “gọn hơn mà khỏe hơn”

##### 03 Thiết kế

Module 4: Kiến trúc Baseline Stacked Bi-LSTM

**Tối ưu mô hình bằng siêu tham số bằng Bayesian Optimization**


**MAE RMSE R²**

Baseline 0.7159 0.9698 0.9067

**Tuned 0.4414 0.6697 0.9555**

MLP 0.9310 1.2968 0.8331

Bi-GRU 0.7551 0.9103 0.9178

LSTM 0.5213 0.7603 0.9426

##### 04 Thực nghiệm

So sánh các mô hình


##### 04 Thực nghiệm

So sánh các mô hình


##### 04 Thực nghiệm

Phân tích lỗi

**Cải thiện đáng kể:**

- Very High (8–9): −46.1% MAE
- Sitting: −31.2% MAE


**Rank Permutation SHAP Correlation RF Surrogate**

1 Heart_Rate Heart_Rate Mood_Score Location

2 Mood_Score Mood_Score Phone_Event Mood_Score

3 Screen_Usage Screen_Usage Heart_Rate Phone_Event

4 Energy_Level Day_of_Week Location Heart_Rate

5 Day_of_Week Hour Screen_Usage Energy_Level

##### 04 Thực nghiệm

Phân tích tầm quan trọng đặc trưng

```
Phản ánh đúng nghiên cứu của Hovsepian: Nhịp tim là chỉ số stress sinh lý quan
trọng nhất, còn tâm trạng (mood) phản ánh trực tiếp trạng thái tâm lý
```

##### 05. Kết luận

Hạn chế

**Dữ liệu mô phỏng Một đối tượng duy nhất**

**Hạn chế về cấu hình**


##### 05. Kết luận

Hướng phát triển

**Thu thập dữ liệu thực tế trên smartphone/wearable Xây dựng hệ thống dự đoán stress thời gian thực**


##### 05 Kết luận

**Kết luận trọng tâm**

```
Khóa luận không đề xuất một mô hình HAR hoàn toàn mới, mà đề xuất một cách khai thác
nhãn hoạt động như ngữ cảnh để dự đoán stress liên tục.
```
```
Đóng góp cốt lõi là Context-Stress Modifier và bộ sinh dữ liệu đa phương thức có truy vết:
paper cung cấp cơ chế/chiều tác động, còn hệ số là heuristic chuẩn hóa cho mô phỏng.
```
```
Kết quả thực nghiệm cho thấy pipeline học được logic context-aware:
Bi-LSTM Tuned đạt R² = 0.9555, MAE = 0.4414 trong thiết lập bán mô phỏng.
Con số này là bằng chứng proof-of-concept, không phải kết luận lâm sàng ngoài thực địa.
```
```
Bài học chính: với bài toán chuỗi stress, thiết kế dữ liệu, chống leakage, ngữ cảnh và tuning
quan trọng không kém việc chọn kiến trúc mô hình phức tạp hơn.
```

# Cảm ơn thầy cô đã lắng nghe!
