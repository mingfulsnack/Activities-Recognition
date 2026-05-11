# KỊCH BẢN NÓI THEO SLIDE BẢO VỆ KHÓA LUẬN (BẢN NÂNG CAO)

Tài liệu này bám theo file slide:
- Doc/LaTeX/defense_slides_research.tex (bản cập nhật)

**Nguyên tắc trình bày:**
- Mỗi slide 50–90 giây
- Mở mỗi slide bằng *vấn đề/câu hỏi*, đóng bằng *insight/bridge* sang slide tiếp
- Trích dẫn con số cụ thể, giải thích công thức khi chiếu
- Giọng điệu: tự tin, học thuật nhưng dễ hiểu

---

## Slide 1 — Trang bìa
"Kính thưa Hội đồng, em xin trình bày khóa luận: Khung Đa Phương Thức Nhận Thức Ngữ Cảnh cho Dự Đoán Mức Độ Stress Liên Tục. Nghiên cứu này giải quyết một bài toán cốt lõi: làm sao để máy tính hiểu được — khi nhịp tim tăng, đó là stress hay chỉ đơn giản là bạn đang chạy bộ?"

## Slide 2 — Nội dung trình bày
"Bài trình bày gồm 5 phần. Em sẽ bắt đầu bằng việc đặt vấn đề và nhấn mạnh khoảng trống nghiên cứu, sau đó trình bày phương pháp đề xuất — đặc biệt là cơ chế Context-Stress Modifier, rồi đến kết quả thực nghiệm với những phát hiện quan trọng, và cuối cùng là đóng góp, hạn chế và hướng phát triển."

## Slide 3 — Kết quả nổi bật (SLIDE MỚI — HOOK)
"Trước khi đi vào chi tiết, em xin giới thiệu 3 kết quả nổi bật nhất. Thứ nhất, mô hình đạt R-squared bằng 0.956, nghĩa là giải thích được 95.6% phương sai stress trên thang 1 đến 9. Thứ hai, cải thiện mạnh nhất ở nhóm stress rất cao — giảm 46.1% sai số — đây là nhóm quan trọng nhất cho cảnh báo sớm. Và thứ ba, mô hình sau tối ưu nhỏ hơn 49% so với baseline nhưng chính xác hơn 23%, cho thấy baseline ban đầu bị over-parameterized. Bài học cốt lõi mà em sẽ chứng minh trong bài này là: tối ưu siêu tham số quan trọng hơn thay đổi kiến trúc."

## Slide 4 — Động lực nghiên cứu
"Stress kéo dài là một trong những vấn đề sức khỏe cộng đồng nghiêm trọng nhất. WHO báo cáo đại dịch COVID-19 đã làm tăng 25% rối loạn lo âu và trầm cảm toàn cầu. Các phương pháp đánh giá truyền thống dựa trên bảng câu hỏi có 3 hạn chế lớn: chủ quan, không liên tục, và gây gián đoạn.

Từ góc nhìn nghiên cứu, em nhận dạng 3 khoảng trống cốt lõi. Một là, đa số nghiên cứu chỉ phân loại stress thành hai hoặc ba mức — trong khi thực tế stress biến thiên liên tục. Hai là, vấn đề nhập nhằng sinh lý — cùng mức nhịp tim 120 bpm nhưng ý nghĩa stress hoàn toàn khác nếu bạn đang chạy bộ so với đang ngồi họp deadline. Và ba là, stress có tính tích lũy theo thời gian mà các mô hình cũ không nắm bắt được. Từ đây, mục tiêu của khóa luận là xây dựng một pipeline context-aware có thể dự đoán stress liên tục trên thang 1 đến 9."

## Slide 5 — Khoảng trống nghiên cứu và đóng góp
"Slide này tóm tắt 4 khoảng trống nghiên cứu và đóng góp tương ứng. Khoảng trống G1: chuyển từ phân loại rời rạc sang hồi quy liên tục. G2: liên kết HAR với stress trong một pipeline thống nhất 5 module thay vì xử lý tách rời. G3: benchmark 5 kiến trúc trên cùng pipeline — điều mà nhiều nghiên cứu chưa thực hiện. Và G4: giải thích kết quả bằng 4 phương pháp khác nhau thay vì chỉ dùng một phương pháp đơn lẻ.

Điều em muốn nhấn mạnh là: logic nghiên cứu xuyên suốt từ câu hỏi khoa học, đến thiết kế phương pháp, đến bằng chứng thực nghiệm, rồi kết luận. Đây không chỉ là triển khai kỹ thuật mà là một quy trình nghiên cứu khoa học."

## Slide 6 — Câu hỏi nghiên cứu và giả thuyết
"Khóa luận đặt 4 câu hỏi nghiên cứu. RQ1 hỏi: việc tích hợp ngữ cảnh hoạt động có giúp diễn giải tín hiệu sinh lý tốt hơn không? RQ2: mô hình chuỗi thời gian có thực sự vượt trội mô hình phi chuỗi? RQ3: Bayesian Optimization có tạo khác biệt đáng kể? Và RQ4: nhóm đặc trưng nào đóng góp chính?

Mỗi câu hỏi đi kèm giả thuyết cụ thể và phương pháp kiểm chứng riêng. Trong cột thứ 3 của bảng, em đã ghi rõ cách kiểm chứng từng câu hỏi. Toàn bộ thiết kế thực nghiệm ở các phần sau đều nhằm trả lời trực tiếp 4 câu hỏi này."

## Slide 7 — Kiến trúc tổng thể 5 module
"Hệ thống gồm 5 module nối tiếp. Module 1 là HAR — nhận dạng 6 hoạt động từ dữ liệu gia tốc kế WISDM với accuracy 96.19%. Module 2 sinh dữ liệu đa phương thức 54,448 mẫu có tích hợp Context-Stress Modifier. Module 3 chọn đặc trưng — giảm từ 44 trường xuống 13 đặc trưng tối ưu. Module 4 là pipeline chống rò rỉ dữ liệu. Và Module 5 là huấn luyện, tối ưu, và so sánh 5 kiến trúc.

Điểm then chốt: nhãn hoạt động từ Module HAR là đầu vào bắt buộc cho Module sinh dữ liệu. Chính liên kết này tạo nên tính nhận thức ngữ cảnh — context-aware — của toàn hệ thống. Mức stress không chỉ phụ thuộc vào chỉ số sinh lý, mà được điều chỉnh theo hoạt động đang thực hiện."

## Slide 8 — Tạo dữ liệu 54,448 mẫu
"Dữ liệu được sinh bán mô phỏng — kết hợp dữ liệu gia tốc kế thực từ WISDM với dữ liệu hành vi mô phỏng theo lịch trình 30 ngày. Mỗi ngày gồm các hoạt động từ 6 giờ sáng đến 22 giờ 30: sinh hoạt ở nhà, đi làm, làm việc, tập thể dục, giải trí. Hệ thống tạo 2 mẫu mỗi phút, khoảng 1,815 mẫu mỗi ngày.

Một câu hỏi Hội đồng có thể đặt ra: tại sao dùng dữ liệu bán mô phỏng? Lý do có 3: một là cho phép kiểm soát biến thực nghiệm có hệ thống — biết chính xác biến nào ảnh hưởng stress; hai là kiểm chứng giả thuyết Context-Stress Modifier trước khi thu thập dữ liệu thực rất tốn kém; và ba là phần gia tốc kế thực từ WISDM đảm bảo tín hiệu sensor chân thực."

## Slide 8A — Lịch trình mô phỏng chi tiết theo khung giờ
"Slide này đi sâu vào câu hỏi thường gặp: dữ liệu một ngày được tạo như thế nào. Em không sinh ngẫu nhiên hoàn toàn, mà dùng lịch trình có cấu trúc theo nhịp sinh hoạt người đi làm văn phòng: sáng ở nhà rồi commute, buổi sáng và chiều làm việc, tối phục hồi, đêm ngủ.

Ý nghĩa của bảng không chỉ là mô tả thời gian, mà là gắn \'pha stress\' vào từng block. Ví dụ 13h đến 17h là giai đoạn áp lực cao nhất ngày; 20h đến 22h30 là pha hồi phục. Nhờ vậy, mô hình học được xu hướng tăng giảm stress theo ngữ cảnh thời gian, không phải chỉ học từ nhiễu ngẫu nhiên.

Về quy mô, với 2 mẫu/phút trong khung thức, mỗi ngày tạo khoảng 1,815 mẫu; 30 ngày cho ra 54,448 mẫu. Đây là mật độ đủ dày để học chuỗi 60 bước mà vẫn giữ được tính liên tục hành vi."

## Slide 8B — Nhiễu và biến thiên: tránh dữ liệu quá sạch
"Nếu dữ liệu mô phỏng quá sạch thì mô hình sẽ học quá dễ và không có giá trị thực tiễn. Vì vậy em thêm 4 lớp biến thiên.

Lớp thứ nhất là life events với xác suất 6 phần trăm mỗi ngày, kéo dài 1 đến 4 ngày, như deadline hay ốm. Lớp thứ hai là daily noise trên các biến sleep, mood, energy, workload để mỗi ngày đều khác nhau. Lớp thứ ba là chu kỳ tuần: stress tăng dần đến thứ 5, thứ 6 rồi giảm cuối tuần. Lớp thứ tư là chu kỳ tháng 28 ngày.

Ở mức sinh lý, nhịp tim có nhiễu đều từ âm 4 đến cộng 4 bpm và vẫn bị ràng buộc trong ngưỡng hợp lý. Stress đầu ra luôn clip trong [1,9] để phù hợp bài toán hồi quy. Cách làm này cân bằng giữa \'đủ thực\' và \'đủ kiểm soát\'."

## Slide 8C — Cách lồng ghép WISDM vào tập dữ liệu cuối cùng
"Đây là slide trọng tâm cho phản biện \'từ WISDM gốc làm sao thành dữ liệu của em\'. Quy trình gồm 5 bước.

Bước 1, nạp WISDM thực: hơn 1 triệu bản ghi gia tốc kế, 36 người dùng, 6 hoạt động. Bước 2, tạo kho mẫu theo từng hoạt động. Bước 3, ScheduleGenerator sinh lịch 30 ngày và phát nhãn hoạt động theo từng timestamp. Bước 4, WisdmDataLoader lấy mẫu gia tốc kế thực đúng hoạt động đó và gán vào Accelerometer X, Y, Z. Bước 5, Orchestrator hợp nhất phần sensor thực với phần sinh lý-hành vi-ngữ cảnh mô phỏng để tạo một dòng dữ liệu hoàn chỉnh 44 trường.

Nói ngắn gọn: hoạt động và accelerometer không phải tự chế, mà đi từ WISDM thật; phần còn lại được mô phỏng có cơ sở để kiểm chứng giả thuyết context-aware. Kết quả HAR kiểm chứng lại trên dữ liệu sinh đạt trung bình 86.2 phần trăm, cho thấy tín hiệu vẫn giữ đặc tính nhận dạng quan trọng."

## Slide 9 — Công thức điều chỉnh nhịp tim
"Theo slide, nhịp tim tại thời điểm t bằng nhịp tim nghỉ, cộng điều chỉnh theo hoạt động, cộng điều chỉnh theo stress, cộng điều chỉnh theo mệt mỏi, cộng nhiễu.

Em giải thích từng thành phần. Nhịp tim nghỉ tính theo công thức Tanaka 2001 dựa trên tuổi và giới tính — khoảng 68 đến 72 bpm cho người trưởng thành. Điều chỉnh theo hoạt động dựa trên giá trị MET từ Ainsworth 2011 — ví dụ Sitting cộng 0 bpm, Walking cộng 18, Jogging cộng 40 bpm. Điều chỉnh theo stress là stress level trừ 4 nhân 3 bpm — vì stress kích hoạt hệ thần kinh giao cảm, giải phóng adrenaline làm tăng nhịp tim. Và điều chỉnh theo mệt mỏi cộng tối đa 5 bpm khi kiệt sức.

Ý nghĩa quan trọng: nhịp tim không phải là giá trị tĩnh — mà là tổng hợp của hoạt động, stress, mệt mỏi và nhiễu sinh lý. Cách mô hình hóa này tạo nền cho Context-Stress Modifier ở slide tiếp theo."

## Slide 10 — Công thức stress đa yếu tố
"Stress nền tại thời điểm t được tính bằng tổng 8 thành phần. Em nhấn mạnh 3 thành phần quan trọng nhất.

Một, delta S theo thời gian — mô phỏng nhịp cortisol hàng ngày, theo nghiên cứu của Chrousos 2009. Cortisol cao nhất buổi sáng, cortisol awakening response, rồi tăng lên đỉnh buổi chiều khi tích lũy mệt mỏi, rồi giảm dần buổi tối.

Hai, delta S theo hoạt động — dựa trên bằng chứng rằng vận động giải phóng endorphin. Jogging giảm 0.8 đơn vị stress, Walking giảm 0.3, theo nghiên cứu Salmon 2001.

Và ba, delta S momentum — hiệu ứng quán tính. Theo lý thuyết Allostatic Load của McEwen 2008, stress có xu hướng duy trì — stress cao trước đó kéo stress hiện tại lên. Hệ số 0.3 đảm bảo quán tính vừa phải, không khống chế hoàn toàn.

Điểm quan trọng: mô hình không học từ một biến đơn lẻ mà từ tương tác đa yếu tố có căn cứ khoa học. Bây giờ, em đến phần cốt lõi nhất."

## Slide 11 — Context-Stress Modifier (SLIDE CỐT LÕI — nói kỹ nhất)
"Đây là đóng góp then chốt của khóa luận, em xin trình bày kỹ.

Sau khi tính được stress nền S base, hệ thống áp dụng thêm Context-Stress Modifier — S modified bằng S base cộng delta nhân A sleep cộng modifier môi trường cộng modifier xã hội. Ý tưởng dựa trên Mô hình Transaction of Stress của Lazarus và Folkman 1984: mức stress không chỉ phụ thuộc kích thích bên ngoài mà còn phụ thuộc đánh giá nhận thức trong bối cảnh cụ thể.

Hãy nhìn bảng minh họa. Cùng sinh lý tương tự nhưng: Sitting tại Work buổi chiều có deadline, delta bằng cộng 2.0 — stress rất cao. Trong khi Jogging tại Outdoor buổi sáng cuối tuần, tuy nhịp tim 140 bpm — rất cao — nhưng delta bằng trừ 1.5, stress thực tế thấp. Đây chính là cách modifier giải quyết nhập nhằng sinh lý.

Thêm nữa, hệ số A sleep — thiếu ngủ không chỉ tăng stress trực tiếp mà khuếch đại phản ứng stress gấp 1.3 lần, nhưng chỉ khi delta dương tức bối cảnh đã gây stress. Điều này phản ánh phát hiện của McEwen 2008 rằng thiếu ngủ tăng phản ứng amygdala với kích thích tiêu cực nhưng không ảnh hưởng kích thích tích cực. Thiết kế này tạo ra dữ liệu phản ánh đúng cơ chế tâm sinh lý, không phải gán nhãn tùy tiện."

## Slide 12 — Pipeline chống rò rỉ dữ liệu
"Quy trình bắt buộc cho chuỗi thời gian là: Split trước rồi mới Encode, Normalize, và tạo Sequence. Tuyệt đối không được đảo thứ tự.

Em chia sẻ một bài học thực tế. Khi thử 17 đặc trưng kỹ thuật bao gồm các rolling features, training loss lên tới 10.23 thay vì 0.92 như kỳ vọng — mô hình hoàn toàn không hội tụ. Nguyên nhân: rolling window tính trên toàn bộ dữ liệu nên rò rỉ thông tin tương lai. Đây là bài toán data leakage mà Kaufman 2012 đã cảnh báo. Giải pháp: loại bỏ hoàn toàn rolling features, giữ 13 đặc trưng sạch.

Điều này cho thấy feature engineering trên chuỗi thời gian phải đặc biệt cẩn thận — một bước tưởng nhỏ có thể phá hỏng toàn bộ tính hợp lệ của thực nghiệm."

## Slide 13 — Thiết lập thực nghiệm
"Toàn bộ thí nghiệm chạy trên CPU với seed cố định 42 để đảm bảo tái lập. Phần mềm gồm Python 3.12.4, TensorFlow 2.16.1, scikit-learn và SHAP. Nguyên tắc quan trọng nhất: tất cả 5 mô hình đều chạy trên cùng dữ liệu, cùng pipeline, cùng callbacks, cùng protocol đánh giá — đây là điều kiện tiên quyết để so sánh công bằng."

## Slide 14 — Kết quả HAR và baseline
"Module HAR đạt accuracy 96.19% trên tập WISDM, vượt mục tiêu 90%. F1 của Jogging là 0.98, Walking 0.97, Sitting và Standing 0.95. Kết quả này đảm bảo nhãn hoạt động đủ tin cậy để đưa vào pipeline stress.

Baseline stress model — Stacked Bi-LSTM với 128 rồi 64 units, dropout 0.3 — đạt R-squared 0.9245 và MAE 0.6855 trên tập test. Đây đã là kết quả tốt, nhưng em nhận thấy dấu hiệu underfitting — dropout cao khiến mô hình chưa sử dụng hết năng lực biểu diễn, và learning rate thấp có thể khiến hội tụ chậm. Đây là lý do em quyết định thực hiện Bayesian Optimization."

## Slide 15 — Tác động của Bayesian Optimization
"Sau 20 trials tối ưu, kết quả rất đáng chú ý. MAE giảm 22.8% — từ 0.686 xuống 0.529. RMSE giảm 14.2%. R-squared tăng lên 0.944. Và đặc biệt: tổng số tham số giảm 48.9% — từ 320 nghìn xuống 164 nghìn.

3 thay đổi quan trọng nhất. Dropout từ 0.3 xuống 0.1 — giải phóng năng lực biểu diễn vì baseline bị underfitting. Learning rate từ 0.001 lên 0.01 — thoát local minima nhanh hơn, kết hợp với ReduceLROnPlateau sẽ tự giảm dần khi cần. Và LSTM units lớp 1 từ 128 xuống 64 — mô hình gọn hơn mà khỏe hơn.

Insight quan trọng: mô hình nhỏ hơn 49% nhưng chính xác hơn 23%. Điều này chứng minh baseline bị over-parameterized — quá nhiều tham số so với độ phức tạp bài toán."

## Slide 16 — So sánh 5 mô hình
"Slide này cho thấy kết quả so sánh 5 kiến trúc trên cùng điều kiện. Bi-LSTM Tuned đạt R-squared 0.956 và MAE 0.441 — tốt nhất trong 5 mô hình. Ngoài ra, so sánh MLP với R-squared 0.833 và Simple LSTM với R-squared 0.943 cho thấy temporal dependency — phụ thuộc thời gian — đóng góp khoảng 11% R-squared. Điều này xác nhận vai trò thiết yếu của mô hình chuỗi so với mô hình phi chuỗi."

## Slide 17 — Bài học quan trọng nhất (SLIDE MỚI)
"Đây là slide em cho là quan trọng nhất trong toàn bài trình bày. Cùng kiến trúc Stacked Bi-LSTM — chỉ khác siêu tham số: Baseline đạt MAE 0.716, nhưng sau tuning MAE giảm xuống 0.441 — cải thiện 38.4%. Đồng thời số tham số giảm gần một nửa.

Thậm chí còn có phát hiện bất ngờ: Simple LSTM chỉ 1 lớp, 1 chiều, 83 nghìn tham số, đạt R-squared 0.943 — tốt hơn Bi-LSTM Baseline 320 nghìn tham số chỉ đạt 0.907! Kiến trúc phức tạp hơn, nhiều tham số hơn nhưng kém hơn — vì siêu tham số không phù hợp.

Chỉ sau khi tuning, Bi-LSTM mới thể hiện đúng tiềm năng vốn có. Bài học rõ ràng: kiến trúc tốt cộng siêu tham số xấu thua kiến trúc đơn giản cộng siêu tham số phù hợp."

## Slide 18 — Phân tích lỗi
"Mô hình tuned không chỉ tốt hơn trung bình mà cải thiện ở đúng chỗ quan trọng nhất. Nhóm stress rất cao 8 đến 9 — nhóm cần cảnh báo sớm — cải thiện mạnh nhất: giảm 46.1% MAE, từ 1.257 xuống 0.677. Baseline luôn dự đoán thấp hơn thực tế ở nhóm này — bias cộng 1.257 — nhưng tuned đã sửa được phần lớn.

Theo hoạt động, Sitting cải thiện 31.2% — đây là hoạt động có biến động stress lớn nhất vì ngồi ở office và ngồi ở nhà rất khác nhau, và tuning giúp mô hình nắm bắt điều đó.

Tổng thể, 82.7% dự đoán chỉ sai dưới 1 điểm, và median absolute error chỉ 0.34 — phần lớn dự đoán lệch khoảng 1/3 điểm trên thang 9."

## Slide 19 — Feature importance đa phương pháp
"Khi phân tích tầm quan trọng đặc trưng bằng 4 phương pháp độc lập — Permutation, SHAP, Correlation, và RF Surrogate — kết quả cho thấy sự đồng thuận rõ ràng. Mood Score và Heart Rate nhất quán đứng top 2 với hạng trung bình 1.75 và 2.25.

Đây phù hợp với y văn: Hovsepian et al. 2015 xác nhận nhịp tim là chỉ số sinh lý stress quan trọng nhất, còn tâm trạng phản ánh trực tiếp trạng thái tâm lý. Ngoài ra, Screen Usage và Phone Event Frequency cũng đóng vai trò quan trọng — gợi mở hướng digital phenotyping: mẫu sử dụng smartphone liên hệ chặt với mức stress."

## Slide 20 — So sánh thứ hạng đặc trưng
"Khi nhìn theo thứ hạng tổng hợp, có một insight thú vị: Location đứng hạng 1 ở RF Surrogate nhưng chỉ hạng 8 ở Permutation và SHAP. Lý do: Decision Tree trong Random Forest khai thác biến categorical rất tốt vì có thể chia dữ liệu trực tiếp theo 6 locations, trong khi LSTM nắm bắt ảnh hưởng phức tạp hơn thông qua interaction với đặc trưng khác. Việc tổng hợp hạng từ nhiều phương pháp — rank aggregation — giúp giảm thiên lệch từ bất kỳ phương pháp đơn lẻ nào, tăng độ tin cậy cho diễn giải."

## Slide 21 — Tổng hợp theo RQ
"Bốn câu hỏi nghiên cứu đều có bằng chứng trả lời. RQ1: context-aware giúp mô hình học theo tổ hợp ngữ cảnh — 5 đặc trưng ngữ cảnh-hành vi nằm trong top 8. RQ2: mô hình chuỗi vượt phi chuỗi — temporal dependency đóng góp 11% R-squared. RQ3: Bayesian Optimization tạo khác biệt lớn — MAE cải thiện 38.4%, tham số giảm 49%. Và RQ4: nhóm sinh lý-hành vi-ngữ cảnh là cốt lõi, đặc biệt Mood Score và Heart Rate.

Em nhấn mạnh: 4 trên 4 câu hỏi đều có bằng chứng thực nghiệm xác nhận. Đây là tính nhất quán khoa học mà em hướng tới từ đầu."

## Slide 22 — Đóng góp học thuật
"Đóng góp được tổ chức theo 3 tầng. Tầng bài toán: chuyển từ phân loại rời rạc sang hồi quy liên tục — biểu diễn cường độ mịn hơn, phù hợp theo dõi và cảnh báo sớm. Tầng phương pháp: đề xuất Context-Stress Modifier theo bộ ba hoạt động nhân vị trí nhân thời gian — giải quyết trực tiếp nhập nhằng sinh lý. Và tầng bằng chứng: benchmark 5 kiến trúc, phân tích đặc trưng 4 hướng, phân tích lỗi đa chiều — tạo chuỗi lập luận khép kín từ câu hỏi nghiên cứu đến kết luận."

## Slide 23 — Hạn chế nghiên cứu
"Em trình bày hạn chế theo khung validity. Construct validity: nhãn stress bán mô phỏng, chưa phải nhãn lâm sàng — R-squared cao phần nào phản ánh tính nhất quán của bộ sinh. Internal validity: đã kiểm soát leakage nhưng chưa có ablation test cho Context-Stress Modifier. External validity: chưa kiểm chứng trên dữ liệu thực đa đối tượng. Conclusion validity: cần bổ sung kiểm định thống kê qua nhiều lần chạy.

Em xin nói thêm: trình bày rõ hạn chế không làm giảm giá trị nghiên cứu — mà giúp xác định chính xác phạm vi áp dụng và hướng kiểm chứng tiếp theo. Đây cũng là tinh thần khoa học mà em theo đuổi."

## Slide 24 — Hướng phát triển
"Ngắn hạn: thu thập dữ liệu thực tế trên smartphone kết hợp EMA, mở rộng benchmark sang Transformer, và thí nghiệm ablation bật tắt Context-Stress Modifier. Trung hạn: triển khai suy luận gần real-time trên mobile, cá nhân hóa mô hình bằng transfer learning, và federated learning cho bảo mật dữ liệu sức khỏe. Dài hạn: mở rộng đa cảm biến sinh học như EDA, HRV, và hướng predict-to-intervene — tự động đề xuất can thiệp dựa trên stress dự đoán."

## Slide 25 — Kết luận
"Tóm lại, khóa luận đã xây dựng thành công 3 thứ. Một: hệ thống context-aware end-to-end từ dữ liệu cảm biến đến dự đoán stress liên tục, với R-squared 0.956 và MAE 0.441 — trung bình mỗi dự đoán chỉ lệch 0.44 điểm trên thang 9.

Hai: Context-Stress Modifier — giải quyết trực tiếp vấn đề nhập nhằng sinh lý. Cùng nhịp tim nhưng ngồi ở office chiều deadline stress rất cao, chạy bộ sáng cuối tuần stress thấp.

Và ba: bài học transferable — tối ưu siêu tham số cải thiện 38.4% MAE, chứng minh rằng tuning thường quan trọng hơn thay đổi kiến trúc. Bài học này áp dụng được cho nhiều bài toán học sâu khác.

Giá trị cốt lõi không chỉ ở metric cao, mà ở tính nhất quán khoa học: từ câu hỏi, đến phương pháp, đến bằng chứng, đến kết luận. Em xin cảm ơn Hội đồng."

## Slide 26 — Q&A
"Em xin cảm ơn Hội đồng và sẵn sàng trao đổi thêm. Các chủ đề em sẵn sàng nhất là: thiết kế Context-Stress Modifier và cơ sở khoa học, tính hợp lệ của dữ liệu bán mô phỏng so với dữ liệu thực tế, kế hoạch kiểm chứng trên dữ liệu thực địa, và lý do tại sao tuning lại quan trọng hơn kiến trúc."

---

# PHỤ LỤC: CHUẨN BỊ CÂU HỎI PHẢN BIỆN

## Q1: "Dữ liệu mô phỏng thì kết quả có đáng tin không?"
"Dạ, đây là hạn chế lớn nhất mà em nhận thức rõ. Tuy nhiên, dữ liệu bán mô phỏng phục vụ mục đích cụ thể: kiểm chứng giả thuyết Context-Stress Modifier trong môi trường kiểm soát được. Phần gia tốc kế là dữ liệu thực từ WISDM — 1 triệu bản ghi từ 36 người dùng. Phần hành vi mô phỏng dựa trên các nghiên cứu Chrousos 2009, McEwen 2008, Lazarus 1984. Hướng tiếp theo bắt buộc là kiểm chứng trên dữ liệu thực tế."

## Q2: "Tại sao Simple LSTM lại tốt hơn Bi-LSTM Baseline?"
"Đây là phát hiện quan trọng. Baseline Bi-LSTM có 320K tham số với dropout 0.3 — quá nhiều parameters bị tắt khi huấn luyện, gây underfitting. Simple LSTM chỉ 83K tham số, dropout nhẹ hơn, nên fit dữ liệu tốt hơn. Sau khi tuning — giảm dropout xuống 0.1 và giảm LSTM units — Bi-LSTM mới thể hiện đúng tiềm năng: R-squared 0.956 so với Simple LSTM 0.943. Bài học: kiến trúc tốt cần siêu tham số phù hợp."

## Q3: "Context-Stress Modifier có phải hand-crafted rules không?"
"Đúng là modifier dựa trên bảng tra cứu được thiết kế thủ công, nhưng giá trị delta được xây dựng dựa trên lý thuyết stress đã thiết lập: mô hình Lazarus-Folkman 1984 về cognitive appraisal, mô hình Demand-Control của Karasek 1979 về stress nghề nghiệp, và bằng chứng sinh lý từ Chrousos 2009 về cortisol. Hướng phát triển là học modifier từ dữ liệu thực thay vì hand-crafted."

## Q4: "R² = 0.956 có quá cao không? Có overfitting không?"
"Pipeline đã kiểm soát leakage rất chặt: Split trước mọi bước, fit encoder/scaler chỉ trên train, seed cố định. R² cao phần nào phản ánh tính nhất quán của mô hình sinh dữ liệu — đây là hạn chế em đã nêu rõ. Tuy nhiên, phân tích lỗi cho thấy mô hình vẫn gặp khó ở nhóm stress trung bình 4–5 (MAE = 0.751) — nếu overfitting, mọi nhóm sẽ đều tốt. Validation loss liên tục giảm theo training loss, không có dấu hiệu overfitting rõ ràng."

## Q5: "Tại sao không so sánh với Transformer?"
"Transformer chưa nằm trong phạm vi nghiên cứu chính vì 2 lý do: (1) tài nguyên CPU giới hạn — Transformer yêu cầu GPU và nhiều trial hơn, (2) trọng tâm là chứng minh giá trị context-aware và tầm quan trọng của tuning, không phải tìm kiến trúc tốt nhất. Tuy nhiên, em đã đưa Transformer vào hướng phát triển ngắn hạn."

## Q6: "Cụ thể em lồng ghép WISDM vào dữ liệu sinh như thế nào?"
"Quy trình của em là \'schedule-driven sampling\'. Trước hết, em tách WISDM thành các kho mẫu theo từng hoạt động. Sau đó, lịch trình 30 ngày sinh nhãn hoạt động theo từng timestamp. Với mỗi timestamp, WisdmDataLoader lấy mẫu gia tốc kế thực tương ứng hoạt động đó để điền vào Accelerometer X, Y, Z. Cuối cùng Orchestrator ghép thêm các biến physiological, behavioral và contextual để tạo thành 44 trường.

Vì vậy, thành phần accelerometer trong dữ liệu cuối không phải tín hiệu giả lập. Nó là tín hiệu thực được lấy từ WISDM theo logic hoạt động của lịch trình. Đây là điểm giúp dữ liệu vừa có tính chân thực sensor, vừa đủ linh hoạt để kiểm chứng giả thuyết Context-Stress Modifier." 
