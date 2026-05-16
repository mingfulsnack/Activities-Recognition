# KỊCH BẢN NÓI THEO SLIDE BẢO VỆ KHÓA LUẬN

Tài liệu này bám theo file `Doc/LaTeX/defense_slides_research.tex` sau khi chỉnh lại nội dung bảo vệ.

Nguyên tắc nói:
- Mỗi slide khoảng 40-70 giây; các slide công thức có thể nói 90 giây nếu hội đồng quan tâm.
- Luôn phân biệt rõ: kết quả metric thuộc phạm vi dữ liệu bán mô phỏng, không phải kết luận lâm sàng ngoài thực địa.
- Khi nói về công thức Module 2, dùng câu chốt: "Đây là heuristic có neo nghiên cứu, không phải công thức y sinh lấy nguyên văn từ paper."
- Nếu bị hỏi sâu, trả lời theo cấu trúc: paper hỗ trợ cơ chế/chiều tác động/khoảng hợp lý; hệ số cụ thể là tham số mô phỏng được chuẩn hóa.

---

## Slide 1 — Trang bìa

"Kính thưa Hội đồng, em xin trình bày khóa luận: Khung Đa Phương Thức Nhận Thức Ngữ Cảnh cho Dự Đoán Mức Độ Stress Liên Tục.

Bài toán em tập trung không chỉ là dự đoán stress, mà là diễn giải đúng ý nghĩa của tín hiệu sinh lý theo bối cảnh. Ví dụ cùng một nhịp tim cao, hệ thống cần phân biệt được đó là do chạy bộ, do mệt mỏi, hay do stress khi đang làm việc."

## Slide 2 — Nội dung trình bày

"Bài trình bày gồm 5 phần: đặt vấn đề, phương pháp đề xuất, thực nghiệm và kết quả, đóng góp-hạn chế-hướng phát triển, và kết luận.

Trọng tâm của bài là cơ chế Context-Stress Modifier và cách em xây dựng pipeline stress liên tục theo hướng context-aware."

## Slide 3 — Thông điệp chính

"Trước khi đi vào chi tiết, em xin nêu thông điệp chính của khóa luận.

Thứ nhất, mô hình tốt nhất đạt R-squared 0.956 trong thiết lập dữ liệu bán mô phỏng. Con số này cần được hiểu đúng phạm vi: nó phản ánh hiệu quả trong bộ dữ liệu do hệ thống sinh có kiểm soát, không phải hiệu năng lâm sàng ngoài thực địa.

Thứ hai, đóng góp quan trọng hơn là cách đưa ngữ cảnh vào diễn giải stress. Hệ thống không xem HR cao là stress cao một cách máy móc, mà xét thêm hoạt động, vị trí và thời điểm.

Thứ ba, các công thức trong Module 2 là heuristic có neo nghiên cứu. Paper cung cấp cơ chế và chiều tác động; còn hệ số cụ thể được chuẩn hóa cho thang stress 1 đến 9."

## Slide 4 — Động lực nghiên cứu

"Stress kéo dài là vấn đề sức khỏe cộng đồng nghiêm trọng. WHO ghi nhận sau COVID-19, rối loạn lo âu và trầm cảm tăng mạnh trên toàn cầu.

Các cách đánh giá truyền thống như bảng hỏi có hạn chế: chủ quan, không liên tục, và thường làm gián đoạn người dùng. Trong khi đó, smartphone và wearable có thể cung cấp tín hiệu liên tục hơn.

Khoảng trống em tập trung gồm ba điểm: stress thường bị phân loại rời rạc thay vì dự đoán liên tục; tín hiệu sinh lý như nhịp tim bị nhập nhằng; và stress có tính tích lũy theo thời gian."

## Slide 5 — Khoảng trống nghiên cứu

"Slide này ánh xạ khoảng trống nghiên cứu với hướng tiếp cận của khóa luận.

G1 là chuyển từ phân loại stress sang hồi quy liên tục trên thang 1 đến 9. G2 là liên kết HAR với stress trong pipeline thống nhất 4 module. G3 là benchmark 5 kiến trúc trên cùng điều kiện để so sánh công bằng. G4 là giải thích kết quả bằng nhiều phương pháp feature importance và phân tích lỗi.

Logic xuyên suốt là: câu hỏi nghiên cứu dẫn tới thiết kế phương pháp, sau đó được kiểm chứng bằng thực nghiệm."

## Slide 6 — Câu hỏi nghiên cứu và giả thuyết

"Khóa luận đặt 4 câu hỏi nghiên cứu.

RQ1 hỏi liệu context-aware có giúp diễn giải tín hiệu sinh lý tốt hơn không. RQ2 hỏi mô hình chuỗi có vượt mô hình phi chuỗi không. RQ3 kiểm tra tác động của Bayesian Optimization. RQ4 tìm nhóm đặc trưng đóng góp chính.

Mỗi câu hỏi đều có cách kiểm chứng riêng, nên phần thực nghiệm phía sau không chỉ là chạy model, mà là trả lời trực tiếp các giả thuyết đã đặt ra."

## Slide 7 — Kiến trúc tổng thể hệ thống

"Hệ thống gồm 4 module.

Module 1 là HAR, nhận dạng 6 hoạt động từ WISDM. Module 2 sinh dữ liệu đa phương thức 54,448 mẫu, trong đó có nhịp tim, mood, sleep, phone usage, location và stress label. Module 3 gồm chọn đặc trưng và pipeline chống leakage. Module 4 là huấn luyện, tuning, so sánh và giải thích mô hình.

Điểm then chốt là nhãn hoạt động từ HAR đi vào Module 2. Nhờ vậy, stress không chỉ phụ thuộc vào tín hiệu sinh lý, mà được diễn giải theo hoạt động đang diễn ra."

## Slide 8 — Module 1: Kết quả HAR

"Module HAR được huấn luyện trên WISDM v1.1 với hơn 1 triệu bản ghi gia tốc kế và 6 hoạt động.

Kết quả accuracy đạt 96.19%. Các hoạt động như Jogging, Sitting, Standing có F1 rất cao; Upstairs và Downstairs khó hơn do tín hiệu tương tự nhau nhưng vẫn đạt mức chấp nhận được.

Vai trò của module này là cung cấp activity label đủ tin cậy cho phần sinh dữ liệu stress. Nếu không có activity label, hệ thống sẽ rất dễ nhầm HR cao do vận động thành stress cao."

## Slide 9 — Tạo dữ liệu đa phương thức

"Dữ liệu cuối gồm 54,448 mẫu trong 30 ngày, với tần suất khoảng 2 mẫu mỗi phút trong khung thức.

Em dùng dữ liệu bán mô phỏng vì mục tiêu ở giai đoạn này là kiểm chứng cơ chế context-aware trong môi trường có kiểm soát. Phần accelerometer là dữ liệu thực từ WISDM; phần hành vi và sinh lý như mood, sleep, HR, phone usage được mô phỏng có cơ sở.

Điểm cần nói rõ là nhãn stress không phải nhãn lâm sàng. Nó là nhãn mô phỏng rule-based để kiểm chứng pipeline và giả thuyết nghiên cứu."

## Slide 10 — Lịch trình mô phỏng trong 1 ngày

"Slide này giải thích dữ liệu một ngày được tạo như thế nào.

Em không sinh ngẫu nhiên hoàn toàn, mà dùng lịch trình có cấu trúc: sáng ở nhà, commute, làm việc buổi sáng, nghỉ trưa, làm việc buổi chiều, về nhà, tập luyện hoặc nghỉ ngơi, rồi ngủ.

Ý nghĩa của lịch trình là gắn stress với bối cảnh thời gian. Ví dụ buổi chiều ở work thường có áp lực cao hơn, còn buổi tối ở nhà là pha phục hồi. Đây là cách tạo nhịp ngày hợp lý cho dữ liệu."

## Slide 11 — Nhiễu và biến thiên

"Nếu dữ liệu mô phỏng quá sạch, mô hình sẽ học quá dễ và kết quả không có nhiều ý nghĩa. Vì vậy em thêm nhiều lớp biến thiên.

Có life events như deadline, ốm, thi cử; có daily noise cho sleep, mood, energy, workload; có chu kỳ tuần và chu kỳ tháng. Ở mức sinh lý, HR có nhiễu nhỏ và vẫn bị ràng buộc trong giới hạn hợp lý.

Mục tiêu là cân bằng giữa hai yêu cầu: dữ liệu đủ có cấu trúc để kiểm chứng giả thuyết, nhưng không quá sạch đến mức phi thực tế."

## Slide 12 — Từ WISDM gốc đến 44 trường

"Slide này trả lời câu hỏi: WISDM gốc được lồng vào dữ liệu cuối như thế nào.

Quy trình là schedule-driven sampling. Đầu tiên em nạp WISDM thực, sau đó tạo kho mẫu theo từng hoạt động. Lịch trình 30 ngày sinh ra activity label cho từng timestamp. Với mỗi timestamp, WisdmDataLoader lấy mẫu accelerometer thực tương ứng với hoạt động đó.

Vì vậy, accelerometer X, Y, Z trong dữ liệu cuối không phải tín hiệu tự chế. Chúng đến từ WISDM thật; phần mô phỏng là các biến sinh lý-hành vi-ngữ cảnh được ghép thêm để phục vụ bài toán stress."

## Slide 13 — Nguyên tắc suy luận công thức Module 2

"Đây là slide rất quan trọng khi bảo vệ.

Em không khẳng định các công thức là công thức y sinh được trích nguyên văn từ paper. Cách em làm là evidence-based heuristic mapping: dùng paper để xác định cơ chế tác động, chiều ảnh hưởng và khoảng giá trị hợp lý; sau đó chuẩn hóa thành hệ số trong thang stress 1 đến 9.

Ví dụ, paper không nói stress tăng 1 mức thì HR tăng đúng 3 bpm. Paper chỉ hỗ trợ rằng stress kích hoạt hệ giao cảm và HR/HRV liên quan tới stress. Em chọn 3 bpm để stress rất cao chỉ cộng tối đa khoảng 15 bpm, nhỏ hơn nhiều so với Jogging cộng 40 bpm. Nhờ vậy hệ thống không đánh đồng vận động mạnh với stress cao."

## Slide 14 — Mô hình hóa nhịp tim

"Công thức HR gồm HR nghỉ, ảnh hưởng hoạt động, ảnh hưởng stress, ảnh hưởng mệt mỏi và nhiễu.

HRmax được neo theo Tanaka: 208 trừ 0.7 nhân tuổi. Ảnh hưởng hoạt động dựa trên cường độ MET từ Ainsworth: Sitting gần như không tăng HR, Walking tăng vừa, Jogging tăng mạnh.

Phần stress là `(Stress Level - 4) * 3 bpm`. Trừ 4 vì 4 là mốc gần trung tính trên thang 1-9; nhân 3 vì muốn tác động stress vừa đủ rõ nhưng không lấn át hoạt động. Đây là tham số mô phỏng, không phải hệ số lâm sàng."

## Slide 15 — Công thức stress đa yếu tố

"Stress base được tạo từ nhiều thành phần: stress nền, thời gian, hoạt động, vị trí, workload, HR, sleep và momentum.

Mỗi thành phần là một modifier chuẩn hóa. Ví dụ time được neo vào nhịp sinh học và lịch làm việc; activity dựa trên hướng tác động của vận động; work/location dựa trên bối cảnh áp lực; sleep và momentum dựa trên allostatic load.

Điểm quan trọng là hệ thống không để một biến đơn lẻ quyết định stress. Stress cuối cùng hình thành từ tương tác đa yếu tố, sau đó còn được điều chỉnh bởi Context-Stress Modifier."

## Slide 16 — Context-Stress Modifier

"Đây là đóng góp cốt lõi của khóa luận.

Theo Lazarus và Folkman, stress không chỉ phụ thuộc kích thích bên ngoài mà còn phụ thuộc cách cá nhân đánh giá kích thích đó trong bối cảnh cụ thể. Áp dụng vào dữ liệu cảm biến: cùng một HR cao có thể có ý nghĩa hoàn toàn khác nhau.

Ví dụ, Jogging ngoài trời sáng cuối tuần có HR 140 nhưng stress có thể thấp. Ngược lại, Sitting ở work buổi chiều có deadline, HR chỉ 95 nhưng stress lại cao. Bảng delta ở đây là lookup heuristic để mã hóa tri thức ngữ cảnh, không phải bảng đo lâm sàng."

## Slide 17 — Pipeline chống rò rỉ dữ liệu

"Với chuỗi thời gian, thứ tự xử lý rất quan trọng: split trước, rồi mới encode, normalize và tạo sequence.

Nếu tạo rolling feature hoặc normalize trên toàn bộ dữ liệu trước khi split, thông tin tương lai có thể rò vào train set. Trong quá trình thử nghiệm, các rolling features đã gây vấn đề nên em loại bỏ và giữ 13 đặc trưng sạch.

Điểm em muốn nhấn mạnh là kết quả model chỉ có ý nghĩa khi pipeline không bị leakage."

## Slide 18 — Thiết lập thực nghiệm

"Toàn bộ mô hình được đánh giá trong cùng điều kiện: cùng dữ liệu, cùng pipeline, cùng callbacks, cùng protocol, seed cố định.

Với bài toán stress hồi quy, em dùng MAE, RMSE và R-squared. Với HAR, em dùng Accuracy và F1.

Thiết lập này giúp việc so sánh 5 kiến trúc công bằng hơn, vì sự khác biệt đến từ mô hình và tuning chứ không phải từ xử lý dữ liệu khác nhau."

## Slide 19 — Stress Baseline

"Baseline là Stacked Bi-LSTM với dropout 0.3, learning rate 0.001 và khoảng 320 nghìn tham số.

Baseline đạt R-squared 0.9245 và MAE 0.6855. Kết quả này tốt, nhưng em thấy có dư địa cải thiện: dropout cao có thể gây underfitting, learning rate thấp làm hội tụ chậm, và số tham số khá lớn.

Từ đó em đặt câu hỏi: nếu không đổi kiến trúc, chỉ tuning siêu tham số thì có cải thiện đáng kể không?"

## Slide 20 — Bayesian Optimization

"Sau Bayesian Optimization, MAE giảm 22.8%, RMSE giảm 14.2%, R-squared tăng lên 0.944, trong khi số tham số giảm gần 49%.

Điều này cho thấy mô hình baseline không thiếu độ phức tạp; vấn đề chính là siêu tham số chưa phù hợp. Dropout giảm từ 0.3 xuống 0.1, learning rate tăng lên 0.01, và số units được điều chỉnh gọn hơn.

Cách nói an toàn là: trong thiết lập thực nghiệm này, tuning tạo cải thiện rõ rệt và giúp mô hình gọn hơn."

## Slide 21 — So sánh 5 mô hình

"Slide này so sánh 5 kiến trúc trên cùng pipeline.

Bi-LSTM Tuned đạt R-squared 0.956 và MAE 0.441, tốt nhất trong bộ dữ liệu bán mô phỏng. So sánh MLP với Simple LSTM cho thấy mô hình chuỗi có lợi thế rõ, vì stress phụ thuộc vào lịch sử gần đây chứ không chỉ trạng thái tại một thời điểm.

Em không trình bày con số này như hiệu năng ngoài đời thật, mà như bằng chứng rằng pipeline context-aware có thể được mô hình chuỗi khai thác hiệu quả."

## Slide 22 — Bài học từ so sánh mô hình

"Bài học chính là: kiến trúc mạnh nhưng siêu tham số không phù hợp có thể thua kiến trúc đơn giản hơn.

Bi-LSTM Baseline nhiều tham số hơn nhưng kém Simple LSTM; sau tuning, Bi-LSTM mới thể hiện đúng tiềm năng. Điều này nhấn mạnh vai trò của Bayesian Optimization và protocol so sánh công bằng.

Em tránh nói tuning luôn quan trọng hơn kiến trúc trong mọi bài toán. Em chỉ kết luận trong phạm vi thực nghiệm này, tuning có tác động rất lớn."

## Slide 23 — Phân tích lỗi

"Mô hình tuned không chỉ cải thiện trung bình mà còn cải thiện ở nhóm quan trọng: stress rất cao từ 8 đến 9.

MAE ở nhóm này giảm 46.1%, tức mô hình giảm sai số ở vùng có ý nghĩa cảnh báo sớm. Theo hoạt động, Sitting cũng cải thiện mạnh vì Sitting là hoạt động nhập nhằng: ngồi ở nhà và ngồi ở work có ý nghĩa stress rất khác nhau.

Phân tích lỗi giúp chứng minh mô hình không chỉ đẹp ở metric tổng quát, mà cải thiện ở đúng vùng cần quan tâm."

## Slide 24 — Feature importance

"Em dùng 4 phương pháp feature importance để tránh phụ thuộc vào một cách giải thích duy nhất.

Mood Score và Heart Rate nhất quán nằm trong top 2. Điều này hợp lý vì mood phản ánh trạng thái tâm lý, còn HR là biosignal phổ biến trong stress detection. Các đặc trưng smartphone như screen usage và phone event cũng quan trọng, phù hợp với hướng digital phenotyping.

Tuy nhiên cần nhớ: vì dữ liệu là bán mô phỏng, feature importance phản ánh logic của bộ sinh dữ liệu và cách model học logic đó."

## Slide 25 — So sánh thứ hạng đặc trưng

"Ở slide này, em nhấn mạnh vì sao dùng nhiều phương pháp giải thích là cần thiết.

Location có thể rất quan trọng trong Random Forest surrogate nhưng không nhất thiết đứng đầu trong SHAP hoặc Permutation của LSTM. Lý do là mỗi phương pháp nhìn feature theo một cách khác nhau.

Việc tổng hợp thứ hạng giúp giảm thiên lệch từ từng phương pháp riêng lẻ và cho cái nhìn ổn định hơn."

## Slide 26 — Tổng hợp theo câu hỏi nghiên cứu

"Bốn câu hỏi nghiên cứu đều có bằng chứng trả lời.

RQ1: context-aware có vai trò vì các đặc trưng ngữ cảnh-hành vi xuất hiện trong nhóm quan trọng. RQ2: mô hình chuỗi vượt mô hình phi chuỗi. RQ3: Bayesian Optimization cải thiện rõ. RQ4: mood, HR và nhóm sinh lý-hành vi-ngữ cảnh là nhóm đặc trưng cốt lõi.

Đây là slide nối phần kết quả với phần đóng góp."

## Slide 27 — Tóm tắt kết quả sau thực nghiệm

"Sau khi trình bày chi tiết, em tóm tắt lại ba kết quả chính.

Một là Bi-LSTM Tuned tốt nhất trong bộ dữ liệu bán mô phỏng. Hai là cải thiện mạnh ở nhóm stress rất cao. Ba là mô hình sau tuning gọn hơn nhưng sai số thấp hơn.

Điểm em muốn giữ là không overclaim: metric cao là kết quả trong phạm vi dataset hiện tại; đóng góp bền hơn là phương pháp xây dựng và kiểm chứng pipeline."

## Slide 28 — Đóng góp phương pháp

"Đóng góp được chia thành ba tầng.

Tầng bài toán: chuyển từ phân loại stress rời rạc sang hồi quy liên tục. Tầng dữ liệu và phương pháp: xây dựng bộ dữ liệu bán mô phỏng đa phương thức có thể truy vết, và đề xuất Context-Stress Modifier. Tầng bằng chứng: benchmark 5 kiến trúc, 4 phương pháp giải thích, và phân tích lỗi đa chiều.

Em nhấn mạnh đây là đóng góp phương pháp và proof-of-concept, chưa phải hệ thống đạt chuẩn lâm sàng."

## Slide 29 — Hạn chế nghiên cứu

"Em trình bày hạn chế theo khung validity.

Construct validity: nhãn stress là bán mô phỏng, chưa phải EMA hay nhãn lâm sàng. Modeling validity: hệ số công thức là heuristic có neo nghiên cứu, chưa được học từ dữ liệu stress thực. Internal validity: chưa có ablation đầy đủ cho từng thành phần Context-Stress Modifier. External validity: chưa kiểm chứng trên dữ liệu thực địa đa đối tượng.

Việc nói rõ hạn chế không làm yếu khóa luận; ngược lại, nó xác định chính xác phạm vi đóng góp và hướng phát triển tiếp theo."

## Slide 30 — Hướng phát triển

"Hướng phát triển ngắn hạn là thu thập dữ liệu thực tế smartphone/wearable kết hợp EMA, học hoặc hiệu chỉnh hệ số công thức từ dữ liệu thật, mở rộng benchmark sang Transformer/TCN, và làm ablation cho Context-Stress Modifier.

Trung hạn là triển khai gần real-time, cá nhân hóa mô hình và bảo vệ quyền riêng tư bằng federated learning. Dài hạn là mở rộng cảm biến như EDA, HRV và hướng predict-to-intervene."

## Slide 31 — Kết luận

"Tóm lại, khóa luận đã xây dựng một pipeline context-aware cho dự đoán stress liên tục trong thiết lập bán mô phỏng.

Giá trị chính gồm: hệ thống end-to-end có kết quả tốt trong phạm vi thử nghiệm; Context-Stress Modifier giúp xử lý nhập nhằng sinh lý; và bài học thực nghiệm rằng tuning siêu tham số có thể tạo khác biệt lớn.

Giá trị cốt lõi không chỉ là metric, mà là khung sinh dữ liệu và dự đoán stress minh bạch, có thể truy vết, sẵn sàng được kiểm chứng bằng dữ liệu thực."

## Slide 32 — Q&A

"Em xin cảm ơn Hội đồng đã lắng nghe. Em sẵn sàng trao đổi thêm về cơ sở hệ số heuristic, thiết kế Context-Stress Modifier, tính hợp lệ của dữ liệu bán mô phỏng, và kế hoạch kiểm chứng thực địa."

---

# PHỤ LỤC: CÂU HỎI PHẢN BIỆN DỄ GẶP

## Q1: "Dữ liệu mô phỏng thì kết quả có đáng tin không?"

"Dạ, em xem đây là hạn chế lớn nhất và đã trình bày rõ. Dữ liệu bán mô phỏng không thay thế dữ liệu thực địa, nhưng có vai trò trong giai đoạn proof-of-concept: giúp kiểm soát biến, kiểm chứng Context-Stress Modifier và đánh giá pipeline trong môi trường có thể truy vết. Phần accelerometer là dữ liệu thực từ WISDM; phần stress label là mô phỏng rule-based. Bước tiếp theo bắt buộc là kiểm chứng bằng dữ liệu thực kết hợp EMA."

## Q2: "Các công thức có phải bịa không?"

"Dạ, không phải bịa ngẫu nhiên, nhưng cũng không phải công thức lâm sàng lấy nguyên văn. Em dùng nghiên cứu để xác định cơ chế, chiều tác động và khoảng hợp lý; ví dụ stress kích hoạt hệ giao cảm, thiếu ngủ tăng phản ứng stress, vận động có thể giảm stress. Sau đó em chuẩn hóa thành hệ số trong thang mô phỏng 1-9. Vì vậy đây là heuristic có neo nghiên cứu."

## Q3: "Tại sao stress level lại trừ 4?"

"Dạ, vì thang stress của em là 1-9 và 4 được chọn làm mốc gần trung tính trong mô hình sinh dữ liệu. Khi stress bằng 4 thì thành phần stress không làm tăng hay giảm HR. Khi stress lớn hơn 4 thì HR tăng nhẹ; khi nhỏ hơn 4 thì HR giảm nhẹ. Đây là cách đặt baseline cho công thức tuyến tính."

## Q4: "Tại sao nhân 3 bpm trong công thức HR-stress?"

"Dạ, paper không nói chính xác là 3 bpm cho mỗi mức stress. Paper chỉ hỗ trợ rằng stress có thể làm tăng HR qua hệ thần kinh giao cảm. Em chọn 3 để stress rất cao từ mức 4 lên 9 chỉ cộng tối đa khoảng 15 bpm. Mức này đủ nhìn thấy nhưng vẫn nhỏ hơn tác động của Jogging khoảng 40 bpm, nên mô hình không nhầm vận động mạnh với stress cao."

## Q5: "Tại sao HR cao không đồng nghĩa stress cao?"

"Dạ, HR là tín hiệu không đặc hiệu. HR cao có thể do chạy bộ, leo cầu thang, mệt mỏi, caffeine hoặc stress. Vì vậy hệ thống cần activity và context để diễn giải. HR 140 khi Jogging ngoài trời có thể bình thường; HR 95 khi Sitting ở work với deadline lại có thể đáng chú ý hơn về stress."

## Q6: "Context-Stress Modifier có phải hand-crafted rules không?"

"Dạ, đúng là hiện tại modifier là lookup rule-based. Điểm quan trọng là rule không tùy tiện: nó mã hóa lý thuyết appraisal của Lazarus-Folkman và hướng context-aware stress detection. Hạn chế là hệ số chưa học từ dữ liệu thật; hướng phát triển là thu thập EMA/wearable data để học hoặc hiệu chỉnh các hệ số này."

## Q7: "R-squared 0.956 có quá cao không?"

"Dạ, con số này cao và cần hiểu đúng phạm vi. Nó là kết quả trên dữ liệu bán mô phỏng, nên phản ánh một phần tính nhất quán của bộ sinh dữ liệu. Em không dùng con số này để khẳng định hiệu năng lâm sàng ngoài thực địa. Giá trị của nó là chứng minh pipeline học được logic context-aware trong môi trường kiểm soát."

## Q8: "Mood Score có gây leakage vì liên quan tới stress không?"

"Dạ, Mood Score có tương quan với stress nhưng không phải bản sao trực tiếp của Stress Level. Nó còn phụ thuộc mood nền theo ngày, activity, location, sleep, phone usage và nhiễu. Trong pipeline, leakage được kiểm soát bằng split trước, encoder/scaler chỉ fit trên train, và không dùng thông tin tương lai khi tạo sequence."

## Q9: "Tại sao chưa học hệ số từ dữ liệu thật?"

"Dạ, vì phạm vi khóa luận là xây dựng proof-of-concept từ WISDM và bộ sinh dữ liệu có kiểm soát. Để học hệ số thật cần dữ liệu thực địa có EMA hoặc nhãn stress đáng tin cậy, đồng bộ với wearable/smartphone. Em đã đưa việc học hoặc hiệu chỉnh hệ số từ dữ liệu thật vào hướng phát triển ngắn hạn."

## Q10: "Vậy đóng góp chính là gì nếu dữ liệu chưa phải thật hoàn toàn?"

"Dạ, đóng góp chính không phải là công bố một hệ thống chẩn đoán stress hoàn chỉnh. Đóng góp là khung phương pháp: kết hợp HAR với sinh dữ liệu đa phương thức, đề xuất Context-Stress Modifier để xử lý nhập nhằng sinh lý, xây dựng pipeline chống leakage, và benchmark nhiều mô hình/giải thích trên cùng điều kiện. Đây là nền tảng để kiểm chứng tiếp trên dữ liệu thực."
