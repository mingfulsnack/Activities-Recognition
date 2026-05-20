# KỊCH BẢN THUYẾT TRÌNH THEO `Final Slide.md`

Tài liệu này được viết lại theo slide mới trong `Doc/Final Slide.md`.

Nguyên tắc nói xuyên suốt:
- Không nói "em đề xuất toàn bộ kiến trúc". Nói đúng hơn: "em xây dựng một pipeline tổng thể, trong đó một số thành phần kế thừa từ nghiên cứu trước; phần đề xuất chính là cách ghép ngữ cảnh hoạt động vào dự đoán stress, bộ sinh dữ liệu đa phương thức có truy vết và Context-Stress Modifier."
- Khi nói về công thức Module 2, dùng câu an toàn: "Các hệ số là heuristic có neo nghiên cứu, không phải công thức y sinh/lâm sàng lấy nguyên văn từ paper."
- Khi nói về kết quả `R² = 0.9555`, nhấn mạnh đây là kết quả trong thiết lập dữ liệu bán mô phỏng, có giá trị proof-of-concept, chưa phải kiểm chứng lâm sàng ngoài thực địa.
- Các nguồn nên nhắc miệng: WISDM/Kwapisz et al. cho dữ liệu hoạt động; Lazarus & Folkman cho stress theo bối cảnh; McEwen cho allostatic load; Ainsworth cho MET; Tanaka cho HRmax; Lane, MoodScope, StudentLife cho mobile sensing; Hovsepian/cStress và WESAD cho stress detection bằng tín hiệu sinh lý.

---

## Slide 1 — Trang bìa

"Kính thưa Hội đồng, em xin trình bày khóa luận: Một mô hình đa phương thức cho phát hiện căng thẳng trong hoạt động hằng ngày.

Ý tưởng trung tâm của khóa luận là: stress không thể được hiểu chính xác nếu chỉ nhìn một tín hiệu đơn lẻ như nhịp tim. Cùng một nhịp tim cao có thể là do chạy bộ, leo cầu thang, mệt mỏi, hoặc stress trong công việc. Vì vậy, khóa luận xây dựng một pipeline đa phương thức, trong đó nhãn hoạt động, ngữ cảnh thời gian, vị trí, hành vi sử dụng điện thoại và tín hiệu sinh lý được kết hợp để dự đoán stress liên tục."

## Slide 2 — Nội dung

"Bài trình bày gồm 5 phần.

Phần 1 đặt vấn đề và mục tiêu nghiên cứu. Phần 2 trình bày thiết kế hệ thống và làm rõ đâu là phần kế thừa, đâu là phần em đề xuất. Phần 3 đi vào thiết kế dữ liệu, công thức mô phỏng và pipeline mô hình. Phần 4 trình bày thực nghiệm và kết quả. Cuối cùng là hạn chế, hướng phát triển và kết luận.

Điểm em muốn nhấn mạnh ngay từ đầu là: đóng góp chính không phải là phát minh lại mô hình HAR hay LSTM, mà là xây dựng một cách diễn giải stress có nhận thức ngữ cảnh."

## Slide 3 — Đặt vấn đề

"Stress kéo dài là một vấn đề sức khỏe tinh thần và thể chất nghiêm trọng. WHO ghi nhận sau đại dịch COVID-19, các rối loạn lo âu và trầm cảm tăng đáng kể trên phạm vi toàn cầu. Stress kéo dài có thể liên quan tới bệnh tim mạch, suy giảm miễn dịch và trầm cảm.

Các phương pháp đánh giá truyền thống, ví dụ bảng hỏi hồi cứu, có ba hạn chế: chủ quan, không liên tục và gây gián đoạn. Trong khi đó, smartphone và wearable có khả năng thu thập tín hiệu liên tục hơn, ví dụ hoạt động, vị trí, nhịp tim, giấc ngủ hoặc hành vi sử dụng điện thoại.

Từ đó, mục tiêu của khóa luận là dự đoán stress liên tục trên thang 1 đến 9 bằng một pipeline context-aware có thể giải thích được."

## Slide 4 — Khoảng trống nghiên cứu

"Em xác định bốn khoảng trống chính.

Thứ nhất, nhiều nghiên cứu stress detection xử lý stress như bài toán phân loại rời rạc, ví dụ stress/không stress, trong khi thực tế stress có cường độ biến thiên liên tục.

Thứ hai, nhận diện hoạt động và dự đoán stress thường bị xử lý tách rời. Điều này gây ra vấn đề nhập nhằng sinh lý: HR cao khi chạy bộ không có cùng ý nghĩa với HR cao khi đang ngồi họp deadline.

Thứ ba, nhiều so sánh mô hình chưa thật sự công bằng vì khác pipeline, khác preprocessing hoặc khác cách chia dữ liệu.

Thứ tư, kết quả mô hình thường thiếu phân tích giải thích. Vì vậy khóa luận dùng feature importance và phân tích lỗi để hiểu mô hình học từ đâu."

## Slide 5 — Kiến trúc tổng thể triển khai

"Slide này mô tả kiến trúc triển khai tổng thể, không nên hiểu là toàn bộ đều là phần em đề xuất mới.

Module 1 là HAR Module, dùng dữ liệu WISDM và mô hình Stacked Bi-LSTM để nhận dạng 6 hoạt động. Phần này em kế thừa bài toán và dữ liệu từ hướng nghiên cứu HAR, không nhận đây là đóng góp mới về mô hình HAR.

Module 2 là Data Generation, sinh tập dữ liệu bán mô phỏng đa phương thức, gồm accelerometer thật từ WISDM và các biến mô phỏng như nhịp tim, mood, energy, screen usage, sleep, location và stress.

Module 3 là Feature & Pipeline, xử lý dữ liệu theo thứ tự chống leakage: split theo thời gian, encode, normalize và tạo chuỗi.

Module 4 là Model & Evaluation, huấn luyện, tuning, so sánh 5 kiến trúc và giải thích kết quả.

Câu chốt ở slide này: đây là kiến trúc triển khai; phần đề xuất thực sự nằm ở cách kết hợp các module, đặc biệt là Module 2 và Context-Stress Modifier."

Không nên nói:
"Em đề xuất kiến trúc HAR 4 module hoàn toàn mới."

Nên nói:
"Em xây dựng pipeline tổng thể; trong đó HAR và LSTM là thành phần kế thừa/ứng dụng, còn phần đề xuất là cơ chế context-aware và bộ sinh dữ liệu có truy vết."

## Slide 6 — Phương pháp đề xuất của khóa luận

"Đây là slide quan trọng để tránh hiểu nhầm về chữ 'đề xuất'.

Phần đề xuất thứ nhất là ghép nhãn hoạt động từ HAR vào bài toán stress. Nhãn activity không chỉ là output phụ, mà trở thành ngữ cảnh để diễn giải tín hiệu sinh lý. Đây là điểm giải quyết trực tiếp nhập nhằng: cùng HR cao nhưng nếu activity là Jogging thì không nên kết luận stress cao.

Phần đề xuất thứ hai là bộ sinh dữ liệu đa phương thức có truy vết. Em kết hợp accelerometer thật từ WISDM với các biến mô phỏng như Heart_Rate, Screen_Usage, Mood, Energy, Sleep, Location và Stress_Level. Mỗi công thức đều có nguồn neo lý thuyết, ví dụ Ainsworth cho MET, Tanaka cho HRmax, McEwen cho allostatic load, MoodScope và StudentLife cho mobile sensing.

Phần đề xuất thứ ba là Context-Stress Modifier. Đây là lớp điều chỉnh stress dựa trên tổ hợp Activity × Location × Context. Ý tưởng dựa trên Lazarus & Folkman 1984: stress phụ thuộc vào đánh giá nhận thức trong bối cảnh cụ thể, không chỉ phụ thuộc kích thích bên ngoài.

Phần đề xuất thứ tư là protocol kiểm chứng có kiểm soát: split theo thời gian, chống leakage, benchmark 5 mô hình và giải thích bằng feature importance.

Tóm lại, em không đề xuất từng viên gạch riêng lẻ như WISDM hay Bi-LSTM; em đề xuất cách ghép các viên gạch đó thành một pipeline context-aware có thể truy vết."

## Slide 7 — Module 1: Nhận diện hành động

"Module 1 sử dụng tập WISDM, gồm hơn 1 triệu bản ghi gia tốc kế từ Wireless Sensor Data Mining Lab, với 6 hoạt động: Walking, Jogging, Sitting, Standing, Upstairs và Downstairs.

Mục tiêu của module này là cung cấp nhãn hoạt động đủ tin cậy cho Module 2. Kết quả đạt accuracy 96.19%, cho thấy nhãn hoạt động có thể dùng làm đầu vào ngữ cảnh.

Điểm cần nói rõ: phần HAR là nền tảng kỹ thuật để lấy activity context. Đóng góp chính của khóa luận không nằm ở việc phát minh mô hình HAR mới, mà ở cách dùng nhãn HAR để diễn giải stress."

## Slide 8 — Chọn 13 đặc trưng cho tập dữ liệu

"Tập dữ liệu cuối sử dụng 13 đặc trưng thuộc 5 nhóm.

Nhóm thời gian gồm Hour và Day_of_Week. Nhóm hoạt động và cảm biến gồm Activity và ba trục Accelerometer X, Y, Z. Nhóm sinh lý gồm Heart_Rate. Nhóm hành vi gồm Screen_Usage_Current, Phone_Event_Frequency và Mood_Score. Nhóm ngữ cảnh gồm Location, Energy_Level và Sleep_Duration.

Lý do chọn các nhóm này là chúng đại diện cho ba lớp thông tin quan trọng: cơ thể đang làm gì, môi trường/ngữ cảnh là gì, và trạng thái cá nhân đang như thế nào. Đây cũng là tinh thần của mobile sensing và digital phenotyping."

## Slide 9 — Vì sao dùng dữ liệu bán mô phỏng?

"Dữ liệu stress thực địa có nhãn tin cậy rất khó thu thập, vì cần đồng bộ wearable, smartphone và nhãn tự báo cáo như EMA. Trong phạm vi khóa luận, em dùng dữ liệu bán mô phỏng để kiểm chứng proof-of-concept.

Phần accelerometer là thật từ WISDM. Phần hành vi, sinh lý và stress label được mô phỏng có kiểm soát. Điều này cho phép biết rõ biến nào tác động đến stress, kiểm tra được Context-Stress Modifier, và tránh việc mô hình học từ một bộ dữ liệu hoàn toàn không có cấu trúc.

Tuy nhiên, đây cũng là hạn chế lớn: kết quả chưa thể xem là kiểm chứng lâm sàng. Em trình bày rõ điều này ở phần hạn chế."

## Slide 10 — Lịch trình mô phỏng 30 ngày

"Dữ liệu được sinh theo lịch trình 30 ngày, với các bối cảnh như home, commute, work, gym, social và outdoor.

Trong một ngày, stress có logic theo nhịp sinh hoạt: sáng thức dậy, đi làm, làm việc, nghỉ trưa, làm việc buổi chiều, về nhà, tập luyện hoặc nghỉ ngơi, rồi ngủ. Buổi chiều ở work thường có áp lực cao hơn; buổi tối ở nhà thường là pha phục hồi.

Ngoài lịch cố định, em thêm sự kiện đặc biệt như deadline, ốm hoặc thi cử với xác suất 6% mỗi ngày, cộng thêm chu kỳ tuần/tháng. Mục tiêu là dữ liệu không quá sạch, nhưng vẫn có cấu trúc để mô hình học được."

## Slide 11 — Mô phỏng Energy Level

"Energy_Level được hiểu là latent recovery score, tức điểm phục hồi tiềm ẩn, không phải năng lượng sinh lý đo trực tiếp.

Cơ sở lý thuyết là McEwen về allostatic load: khi cơ thể thiếu phục hồi, gánh nặng stress tích lũy tăng lên. StudentLife cũng cho thấy sleep, workload, activity và trạng thái cá nhân có liên hệ trong dữ liệu smartphone theo thời gian.

Vì vậy Energy_Level được tạo từ nền 0.7, cộng dao động theo chu kỳ, mệt mỏi tích lũy theo tuần và sự kiện. Nó bị clip trong khoảng 0.2 đến 1.0 để tránh giá trị phi thực tế.

Câu cần nhớ: Energy_Level là feature mô phỏng trạng thái phục hồi, không phải đại lượng y sinh đo trực tiếp."

## Slide 12 — Mô hình hóa nhịp tim

"Nhịp tim được tính từ nhiều thành phần: HR nghỉ, ảnh hưởng của hoạt động, ảnh hưởng của stress, ảnh hưởng của fatigue và nhiễu.

HRrest lấy từ hồ sơ người dùng. HRmax được neo theo Tanaka et al. 2001 với công thức 208 - 0.7 × age. Ảnh hưởng hoạt động được quy đổi từ MET theo Ainsworth Compendium of Physical Activities. Ví dụ Jogging có MET cao nên cộng HR lớn hơn Walking hoặc Sitting.

Thành phần stress là `(StressLevel - 4) × 3 bpm`. Paper không cho hệ số đúng 3 bpm. Paper chỉ hỗ trợ cơ chế stress kích hoạt hệ giao cảm, HR/HRV liên quan đến stress. Em chọn 3 bpm để stress rất cao chỉ cộng khoảng 15 bpm, nhỏ hơn tác động vận động mạnh như Jogging."

## Slide 13 — Bảng điều chỉnh HR theo MET

"Bảng này giải thích vì sao mỗi hoạt động cộng một lượng bpm khác nhau.

Sitting gần nghỉ ngơi nên cộng 0 bpm. Standing tăng nhẹ. Walking tăng vừa. Upstairs và Jogging có cường độ cao nên tăng mạnh hơn.

Cơ sở là Ainsworth et al. 2011 về MET Compendium, kết hợp nguyên tắc của American Heart Association rằng nhịp tim tăng theo cường độ vận động. Đây không phải bảng bpm gốc từ Ainsworth; đây là bước quy đổi thực dụng từ MET sang bpm để phục vụ mô phỏng."

## Slide 14 — Công thức stress đa yếu tố

"Stress base được tính từ nhiều thành phần: stress nền, thời gian, hoạt động, vị trí, workload, HR, sleep và momentum.

Điểm quan trọng là stress không được quyết định bởi một feature duy nhất. HR cao chỉ là một tín hiệu; nếu HR cao do Jogging thì ý nghĩa stress khác với HR cao khi Sitting ở Work.

Các thành phần trong công thức đều có nguồn neo: Chrousos cho nhịp cortisol và HPA axis, Salmon cho tác động của vận động lên lo âu/stress, Karasek cho job demand-control, McEwen cho allostatic load, Hovsepian/cStress và WESAD cho stress detection bằng tín hiệu sinh lý."

## Slide 15 — Điều chỉnh theo thời gian

"Điều chỉnh theo thời gian mô phỏng nhịp stress trong ngày.

Cơ sở là Chrousos về hệ HPA và cortisol. Cortisol có nhịp sinh học, thường tăng mạnh quanh giai đoạn thức dậy và thay đổi theo hoạt động trong ngày.

Trong mô phỏng, các khung giờ làm việc được gán hệ số stress cao hơn, đặc biệt buổi chiều vì tích lũy workload và mệt mỏi. Buổi tối và sau 20h giảm dần vì là pha nghỉ ngơi.

Lưu ý: đây là mô phỏng theo xu hướng, không phải đo cortisol thật theo từng người."

## Slide 16 — Điều chỉnh theo hoạt động

"Điều chỉnh theo hoạt động dựa trên ý tưởng rằng vận động thể chất có thể giảm căng thẳng tâm lý, nhưng gắng sức tức thời cũng có thể tạo arousal.

Salmon et al. 2001 tổng quan rằng luyện tập thể chất có thể giảm lo âu, trầm cảm và độ nhạy với stress. Vì vậy các hoạt động như Walking hoặc Jogging trong bối cảnh phù hợp có hệ số giảm stress.

Ngược lại, Upstairs hoặc Downstairs là gắng sức ngắn, dễ làm tăng kích hoạt sinh lý, nên không được xem là thư giãn giống Jogging ngoài trời."

## Slide 17 — Điều chỉnh theo vị trí

"Location được dùng như proxy cho bối cảnh tâm lý.

Work được gán hệ số tăng stress dựa trên mô hình Demand-Control của Karasek: yêu cầu công việc cao và quyền kiểm soát thấp làm tăng stress nghề nghiệp. Commute tăng nhẹ vì bối cảnh giao thông, thời gian gấp và tiếng ồn.

Home, Outdoor, Social và Gym có xu hướng giảm stress tùy ngữ cảnh. Bratman et al. 2015 hỗ trợ rằng môi trường tự nhiên có liên hệ với giảm rumination và phục hồi tinh thần. Social có thể giảm stress nhờ hỗ trợ xã hội, nhưng nếu là conflict thì lại tăng; phần này được xử lý rõ hơn ở Context-Stress Modifier."

## Slide 18 — Điều chỉnh theo HR

"HR được dùng như một tín hiệu phụ cho stress, nhưng không được dùng như kết luận trực tiếp.

Hovsepian et al. trong cStress và các nghiên cứu như WESAD cho thấy HR/HRV là nhóm tín hiệu quan trọng trong nhận diện stress. Tuy nhiên HR không đặc hiệu: HR tăng có thể do vận động, caffeine, mệt mỏi hoặc stress.

Vì vậy trong công thức stress, HR chỉ là một modifier. Ý nghĩa thật sự của HR phải được giải thích cùng activity và context."

## Slide 19 — Sleep và Momentum

"Sleep ảnh hưởng stress vì thiếu ngủ làm giảm khả năng phục hồi và điều hòa cảm xúc. McEwen về allostatic load và Yoo et al. về phản ứng amygdala khi thiếu ngủ là nguồn neo chính.

Momentum phản ánh quán tính stress: stress không biến mất ngay ở bước thời gian tiếp theo. Plarre/cStress cũng mô hình hóa stress như tín hiệu có tính tích lũy và suy giảm theo thời gian.

Hệ số momentum 0.3 được chọn để stress trước đó có ảnh hưởng nhưng không chi phối hoàn toàn hiện tại."

## Slide 20 — Screen Usage

"Screen_Usage_Current được mô phỏng dựa trên mobile sensing.

Lane et al. 2010 tổng quan rằng smartphone có thể ghi nhận app usage, location, activity và giao tiếp để suy luận trạng thái cá nhân. MoodScope của Likamwa et al. cũng dùng smartphone usage pattern để suy luận mood.

Trong công thức, activity quyết định khả năng dùng điện thoại nền. Sitting cao hơn Jogging vì khi chạy bộ khó dùng điện thoại. Location điều chỉnh theo bối cảnh: Home/Work thường cao hơn Gym/Outdoor. Stress chỉ tăng nhẹ qua hệ số 0.15 để tránh biến Screen Usage thành bản sao của Stress_Level."

## Slide 21 — Mood Score

"Mood_Score được neo bởi MoodScope và StudentLife.

MoodScope cho thấy mood có thể được suy luận từ mẫu sử dụng smartphone. StudentLife theo dõi sleep, activity, workload, sociability và mental well-being của sinh viên bằng smartphone sensing.

Trong mô phỏng, mood nền của ngày đặt quanh mức 5 trên thang 1 đến 10. Sau đó mood được điều chỉnh bởi thời gian, hoạt động, vị trí và stress. Stress cao kéo mood xuống theo hệ số 0.3, nhưng không quyết định toàn bộ mood.

Câu trả lời nếu bị hỏi: mood có tương quan với stress, nhưng không phải bản sao của stress, vì nó còn phụ thuộc nhiều yếu tố và nhiễu."

## Slide 22 — Context-Stress Modifier

"Đây là phần cốt lõi nhất.

Theo Lazarus & Folkman 1984, stress phụ thuộc vào cognitive appraisal, tức cách cá nhân đánh giá tình huống trong bối cảnh cụ thể. Vì vậy cùng tín hiệu sinh lý có thể có ý nghĩa stress khác nhau.

Ví dụ Jogging ngoài trời sáng cuối tuần có HR cao nhưng stress thấp. Ngược lại, Sitting ở Work buổi chiều có deadline, HR không quá cao nhưng stress có thể cao.

Context-Stress Modifier mã hóa ý tưởng này bằng bảng delta theo Activity × Location × Context. Bảng này là heuristic có cơ sở, không phải bảng đo lâm sàng."

## Slide 23 — Các modifier phụ trong Context-Stress

"Ngoài delta theo activity-location-context, hệ thống còn có sleep amplification, environment modifier và social modifier.

Sleep amplification dựa trên McEwen và Yoo: thiếu ngủ làm tăng phản ứng với tác nhân stress. Environment modifier dựa trên Bratman: môi trường tự nhiên/yên tĩnh có tác dụng phục hồi. Social modifier dựa trên Lazarus-Folkman và StudentLife: quan hệ xã hội và sociability ảnh hưởng đến trạng thái tinh thần.

Tất cả modifier này là mức điều chỉnh nhỏ để mô phỏng bối cảnh, không phải công thức lâm sàng."

## Slide 24 — Module 3: Tiền xử lý dữ liệu

"Pipeline tiền xử lý gồm bốn bước: Sequential Split, Label Encoding, Standard Normalization và Sequence.

Điểm quan trọng nhất là split theo thời gian trước, không shuffle. Train chiếm 70%, validation 15%, test 15%. Với chuỗi thời gian, nếu shuffle hoặc fit scaler trên toàn bộ dữ liệu trước khi split thì rất dễ rò rỉ thông tin tương lai.

Đây là phần bảo vệ tính hợp lệ nội bộ của thực nghiệm."

## Slide 25 — Encode, Normalize và tạo Sequence

"Hai biến phân loại Activity và Location được mã hóa bằng LabelEncoder. Encoder chỉ fit trên train, sau đó transform cho validation và test.

Normalization cũng chỉ dùng mean và standard deviation của train. Val/test không được fit lại.

Sau đó dữ liệu được biến thành chuỗi độ dài 60. Mỗi cửa sổ gồm 60 vector liên tiếp, mỗi vector có 13 feature, và mô hình dự đoán stress ở bước tiếp theo.

Đây là lý do mô hình LSTM/Bi-LSTM có thể khai thác phụ thuộc thời gian."

## Slide 26 — Baseline Stacked Bi-LSTM

"Baseline là Stacked Bi-LSTM với hai lớp LSTM hai chiều, dropout 0.3 và các lớp Dense phía sau.

Mô hình có khoảng 320 nghìn tham số. Kết quả baseline đã tốt, nhưng phân tích cho thấy dropout 0.3 có thể quá cao, learning rate 0.001 hội tụ chậm, và mô hình có dấu hiệu over-parameterized.

Vì vậy em dùng Bayesian Optimization để tìm bộ siêu tham số phù hợp hơn."

## Slide 27 — Bayesian Optimization

"Sau tuning, số units lớp đầu giảm từ 128 xuống 64, dropout giảm từ 0.3 xuống 0.1, dense_units tăng lên 128 và learning rate tăng lên 0.01.

Kết quả là số tham số giảm gần 48.9%, nhưng hiệu suất tốt hơn. Điều này cho thấy vấn đề không phải cứ mô hình lớn hơn là tốt hơn; mô hình cần siêu tham số phù hợp với dữ liệu.

Em trình bày kết quả tuning như một bài học thực nghiệm, không khẳng định Bayesian Optimization là phương pháp mới do em phát minh."

## Slide 28 — So sánh mô hình

"Năm mô hình được so sánh trên cùng pipeline: MLP, Simple LSTM, Bi-LSTM Baseline, Bi-GRU và Bi-LSTM Tuned.

Bi-LSTM Tuned đạt MAE 0.4414, RMSE 0.6697 và R² 0.9555. Simple LSTM cũng đạt kết quả tốt hơn MLP, cho thấy temporal dependency có vai trò quan trọng trong bài toán stress.

Câu cần nhấn mạnh: kết quả này nằm trong thiết lập bán mô phỏng. Nó chứng minh pipeline học được logic context-aware, chưa chứng minh hiệu năng thực địa."

## Slide 29 — Phân tích lỗi

"Phân tích lỗi cho thấy tuned model cải thiện mạnh ở nhóm Very High stress và ở hoạt động Sitting.

Very High stress quan trọng vì đây là nhóm có ý nghĩa cảnh báo sớm. Sitting cũng quan trọng vì đây là activity nhập nhằng: ngồi ở nhà buổi tối có thể thư giãn, còn ngồi ở work với deadline có thể stress cao.

Kết quả này ủng hộ vai trò của context-aware: mô hình không chỉ giảm lỗi trung bình, mà cải thiện ở các vùng ngữ cảnh khó."

## Slide 30 — Feature Importance

"Feature importance được phân tích bằng nhiều phương pháp: Permutation, SHAP, Correlation và RF Surrogate.

Heart_Rate và Mood_Score nhất quán nằm trong nhóm đặc trưng quan trọng nhất. Điều này phù hợp với y văn: Hovsepian/cStress và WESAD cho thấy HR/HRV là biosignal quan trọng cho stress; MoodScope và StudentLife hỗ trợ liên hệ giữa mood, phone usage, sleep, workload và trạng thái cá nhân.

Screen_Usage và Energy_Level cũng xuất hiện trong nhóm quan trọng, củng cố hướng digital phenotyping."

## Slide 31 — Hạn chế

"Khóa luận có ba hạn chế chính.

Thứ nhất, dữ liệu stress là bán mô phỏng, chưa phải dữ liệu thực địa có nhãn EMA hoặc lâm sàng. Vì vậy metric cao có thể phản ánh một phần tính nhất quán của bộ sinh dữ liệu.

Thứ hai, dữ liệu mô phỏng hiện thiên về một hồ sơ người dùng/kịch bản sinh hoạt, nên chưa đại diện nhiều nhóm tuổi, nghề nghiệp và lối sống.

Thứ ba, tài nguyên tính toán còn hạn chế; chưa mở rộng đầy đủ sang Transformer, TCN hoặc nhiều lần chạy thống kê."

## Slide 32 — Hướng phát triển

"Hướng phát triển quan trọng nhất là thu thập dữ liệu thực tế trên smartphone/wearable kết hợp EMA. Khi có dữ liệu thật, các hệ số heuristic có thể được học hoặc hiệu chỉnh thay vì đặt thủ công.

Tiếp theo là triển khai dự đoán gần thời gian thực, mở rộng cảm biến như EDA, HRV, sleep tracker và cá nhân hóa mô hình theo từng người.

Ngoài ra, cần làm ablation study: bật/tắt Context-Stress Modifier để đo trực tiếp đóng góp của phần đề xuất."

## Slide 33 — Kết luận

"Kết luận trọng tâm của khóa luận gồm ba ý.

Thứ nhất, khóa luận không đề xuất một mô hình HAR hoàn toàn mới. HAR là thành phần nền để lấy activity context. Phần đề xuất là cách dùng activity context để diễn giải stress.

Thứ hai, đóng góp cốt lõi là Context-Stress Modifier và bộ sinh dữ liệu đa phương thức có truy vết. Các paper cung cấp cơ chế, chiều tác động và khoảng hợp lý; còn hệ số là heuristic chuẩn hóa cho mô phỏng.

Thứ ba, kết quả thực nghiệm cho thấy pipeline học được logic context-aware: Bi-LSTM Tuned đạt R² = 0.9555 và MAE = 0.4414 trong thiết lập bán mô phỏng. Đây là bằng chứng proof-of-concept, không phải kết luận lâm sàng.

Bài học chính là: với bài toán stress theo chuỗi thời gian, thiết kế dữ liệu, chống leakage, ngữ cảnh và tuning quan trọng không kém việc chọn mô hình phức tạp hơn."

## Slide 34 — Cảm ơn

"Em xin cảm ơn Hội đồng đã lắng nghe. Em sẵn sàng trao đổi thêm về ba điểm: cơ sở lý thuyết của các công thức heuristic, thiết kế Context-Stress Modifier, và kế hoạch kiểm chứng bằng dữ liệu thực địa."

---

# BẢN ĐỒ NGUỒN CẦN NHỚ KHI BỊ HỎI

- WISDM / Kwapisz et al. (2011): nguồn dữ liệu gia tốc kế thật cho 6 hoạt động hằng ngày; hỗ trợ phần HAR và activity context, không hỗ trợ trực tiếp stress.
- Hochreiter & Schmidhuber / LSTM và Bi-LSTM: nền tảng mô hình chuỗi; hỗ trợ lựa chọn kiến trúc, không phải đóng góp mới của khóa luận.
- Lane et al. (2010): mobile sensing cho thấy smartphone có thể ghi nhận activity, location, app usage và hành vi; hỗ trợ Screen Usage và digital phenotyping.
- Likamwa et al. / MoodScope: dùng smartphone usage pattern để suy luận mood; hỗ trợ ý tưởng Mood_Score liên hệ phone usage/hành vi.
- Wang et al. / StudentLife: theo dõi sleep, activity, workload, sociability và mental well-being bằng smartphone sensing; hỗ trợ mood, energy, sleep/workload và social context.
- Tanaka et al. (2001): công thức HRmax theo tuổi; hỗ trợ phần giới hạn sinh lý nhịp tim.
- Ainsworth et al. (2011): MET Compendium; hỗ trợ xếp hạng cường độ hoạt động, từ đó quy đổi heuristic sang `Delta HR_activity`.
- Chrousos (2009): HPA axis/cortisol; hỗ trợ hướng tác động của thời gian trong ngày và stress physiology.
- Salmon (2001): vận động thể chất liên quan giảm lo âu/stress sensitivity; hỗ trợ `Delta S_activity`.
- Karasek (1979): Demand-Control model; hỗ trợ work context làm tăng stress.
- Bratman et al. (2015): môi trường tự nhiên liên quan giảm rumination/phục hồi tinh thần; hỗ trợ outdoor/environment modifier.
- McEwen (2008): allostatic load; hỗ trợ sleep, fatigue, recovery và stress tích lũy.
- Yoo et al. (2007): thiếu ngủ làm tăng phản ứng cảm xúc tiêu cực/amygdala; hỗ trợ sleep amplification.
- Hovsepian/cStress và WESAD: stress detection bằng tín hiệu sinh lý như HR/HRV; hỗ trợ HR là biosignal hữu ích nhưng không đặc hiệu.
- Lazarus & Folkman (1984): Transactional Model of Stress/cognitive appraisal; nguồn lý thuyết quan trọng nhất cho Context-Stress Modifier.

Một câu chốt rất nên thuộc:
"Các nghiên cứu trên không cho trực tiếp hệ số trong công thức. Chúng cung cấp cơ chế, chiều tác động và khoảng hợp lý; còn hệ số cụ thể là heuristic chuẩn hóa cho mô hình mô phỏng."

---

# CÂU HỎI PHẢN BIỆN CẦN CHUẨN BỊ

## Q1: "Vậy phần nào thật sự là đề xuất của em?"

"Dạ, phần đề xuất của em không phải là mô hình HAR hay LSTM riêng lẻ. Các phần đó là thành phần kế thừa/ứng dụng. Phần em đề xuất là cách ghép HAR vào stress prediction như một nguồn context, xây dựng bộ sinh dữ liệu đa phương thức có truy vết, và thiết kế Context-Stress Modifier để xử lý nhập nhằng sinh lý."

## Q2: "Tại sao không gọi toàn bộ kiến trúc là đề xuất?"

"Dạ, vì như vậy sẽ không chính xác. Toàn bộ kiến trúc là pipeline triển khai. Trong pipeline đó, WISDM, HAR, Bi-LSTM, Bayesian Optimization là các thành phần đã có trong y văn hoặc kỹ thuật học máy. Đóng góp của khóa luận là cách tích hợp các thành phần này vào bài toán stress context-aware."

## Q3: "Các hệ số trong công thức có phải lấy nguyên văn từ paper không?"

"Dạ không. Paper không cho các hệ số đúng như 0.15, 0.3 hay 3 bpm. Paper hỗ trợ cơ chế và chiều tác động, ví dụ stress làm tăng kích hoạt hệ giao cảm, thiếu ngủ tăng phản ứng stress, vận động có thể giảm stress. Em chuẩn hóa các tác động đó về thang mô phỏng 1-9. Vì vậy các hệ số là evidence-based heuristic, không phải công thức lâm sàng."

## Q4: "Tại sao dữ liệu mô phỏng vẫn có giá trị khoa học?"

"Dạ, dữ liệu bán mô phỏng có giá trị trong giai đoạn proof-of-concept vì nó cho phép kiểm soát biến, biết rõ cơ chế sinh nhãn và kiểm chứng giả thuyết Context-Stress Modifier. Tuy nhiên em không xem nó là thay thế dữ liệu thực địa. Hướng phát triển bắt buộc là thu thập EMA/wearable data để kiểm chứng ngoài đời thực."

## Q5: "R² = 0.9555 có quá cao không?"

"Dạ, con số này cao vì mô hình được đánh giá trên bộ dữ liệu bán mô phỏng có cấu trúc. Em không dùng con số này để khẳng định hiệu năng lâm sàng. Ý nghĩa của nó là mô hình học được logic context-aware trong môi trường kiểm soát. Khi chuyển sang dữ liệu thực, kết quả có thể thấp hơn và cần external validation."

## Q6: "Tại sao HR cao không đồng nghĩa stress cao?"

"Dạ, HR không đặc hiệu cho stress. HR cao có thể do chạy bộ, leo cầu thang, caffeine, mệt mỏi hoặc stress. Vì vậy cần activity và context. HR 140 khi Jogging ngoài trời có thể bình thường, nhưng HR 95 khi Sitting ở Work với deadline lại có thể đáng chú ý hơn về stress."

## Q7: "Mood Score có gây data leakage vì liên quan stress không?"

"Dạ, Mood Score có tương quan với stress nhưng không phải bản sao trực tiếp của Stress_Level. Nó còn phụ thuộc mood nền theo ngày, activity, location, sleep, phone usage và nhiễu. Pipeline cũng split theo thời gian trước, encoder/scaler chỉ fit trên train, nên không dùng thông tin tương lai."

## Q8: "Nếu được làm tiếp, bước quan trọng nhất là gì?"

"Dạ, bước quan trọng nhất là thu thập dữ liệu thực tế có EMA hoặc self-report đồng bộ với smartphone/wearable. Khi có dữ liệu thật, em có thể học hoặc hiệu chỉnh các hệ số hiện đang là heuristic, kiểm chứng Context-Stress Modifier bằng ablation, và đánh giá khả năng tổng quát hóa trên nhiều người dùng."
