# Digital-Time-Reader


giải quyết vấn đề cắt:

Problem: cần đọc vào từng pixel, nếu chia theo chiều ngang dễ bị sai do phần nền thừa 
=> cắt theo vùng sáng (pixel trắng)
Problem: ảnh nhòe, có những vùng 2 số bị đè lên nhau, nếu chỉ cắt theo vùng sáng, sẽ có nhưng vùng là 2 chữ số
=> đo độ dài của vùng nếu width = 0.7 height => bị nhòe => cắt làm 2 vùng
Problem: Vấn đề 2 chưa được giải quyết triệt để do thuật toán chưa bao quát được hết các case, vần có những ảnh chỉ cắt được 3 vùng
=> Chỉ giữ lại các ảnh được cắt thành 4 vùng. Cắt nhiều hoặc ít hơn (3 hoặc 5) thì bỏ qua.

