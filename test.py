# DECISION TREE FOR DIABETES PREDICTION - COMPLETE CODE

# ======================
# 1. IMPORT LIBRARIES
# ======================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.tree import export_graphviz
import graphviz
warnings.filterwarnings('ignore')

# ======================
# 2. LOAD AND EXPLORE DATA
# ======================
# Load dataset từ sklearn (hoặc từ file CSV)
from sklearn.datasets import load_breast_cancer  # Hoặc dùng diabetes dataset

# Hoặc dùng dataset Diabetes (phổ biến hơn)
# Tải dataset từ URL
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, names=columns)

print("="*50)
print("THÔNG TIN DATASET")
print("="*50)
print(f"Kích thước dataset: {df.shape}")
print(f"\n5 dòng đầu tiên:")
print(df.head())
print(f"\nThông tin các cột:")
print(df.info())
print(f"\nThống kê mô tả:")
print(df.describe())
print(f"\nKiểm tra giá trị thiếu:")
print(df.isnull().sum())

# ======================
# 3. DATA PREPROCESSING
# ======================
print("\n" + "="*50)
print("TIỀN XỬ LÝ DỮ LIỆU")
print("="*50)

# Kiểm tra giá trị 0 không hợp lệ (trong medical data, 0 có thể là missing)
# Với Glucose, BloodPressure, SkinThickness, Insulin, BMI - giá trị 0 là không hợp lý
cols_to_check = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in cols_to_check:
    df[col] = df[col].replace(0, np.nan)

print(f"Số giá trị thiếu sau khi thay thế 0:")
print(df.isnull().sum())

# Điền giá trị thiếu bằng median
for col in cols_to_check:
    df[col].fillna(df[col].median(), inplace=True)

print(f"\nĐã điền giá trị thiếu bằng median")

# Phân tích phân phối
plt.figure(figsize=(12, 8))
for i, col in enumerate(columns[:-1], 1):
    plt.subplot(3, 3, i)
    sns.histplot(df[col], kde=True)
    plt.title(f'Phân phối của {col}')
plt.tight_layout()
plt.savefig('data_distribution.png', dpi=300)
plt.show()

# ======================
# 4. PREPARE DATA FOR MODELING
# ======================
# Tách features và target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

print("\n" + "="*50)
print("PHÂN TÍCH LỚP MỤC TIÊU")
print("="*50)
print(y.value_counts())
print(f"\nTỷ lệ lớp 0 (Không tiểu đường): {sum(y==0)/len(y)*100:.2f}%")
print(f"Tỷ lệ lớp 1 (Có tiểu đường): {sum(y==1)/len(y)*100:.2f}%")

# Chuẩn hóa dữ liệu (Decision Tree không bắt buộc nhưng có thể thử)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Chia train/test (70/30 hoặc 80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nKích thước tập train: {X_train.shape}")
print(f"Kích thước tập test: {X_test.shape}")

# ======================
# 5. TRAIN DECISION TREE MODEL
# ======================
print("\n" + "="*50)
print("HUẤN LUYỆN MÔ HÌNH")
print("="*50)

# Tạo và huấn luyện mô hình
dt_model = DecisionTreeClassifier(
    criterion='gini',       # hoặc 'entropy'
    max_depth=4,           # Giới hạn độ sâu để tránh overfitting
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

dt_model.fit(X_train, y_train)

# Dự đoán
y_pred = dt_model.predict(X_test)
y_pred_train = dt_model.predict(X_train)

# ======================
# 6. EVALUATE MODEL
# ======================
print("\n" + "="*50)
print("ĐÁNH GIÁ MÔ HÌNH")
print("="*50)

# Tính các chỉ số
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Độ chính xác trên tập train: {train_accuracy:.4f}")
print(f"Độ chính xác trên tập test:  {test_accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

# Báo cáo chi tiết
print("\n" + "-"*50)
print("BÁO CÁO PHÂN LOẠI CHI TIẾT:")
print("-"*50)
print(classification_report(y_test, y_pred, target_names=['Không TD', 'Có TD']))

# Ma trận nhầm lẫn
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Không TD', 'Có TD'],
            yticklabels=['Không TD', 'Có TD'])
plt.title('MA TRẬN NHẦM LẪN (Confusion Matrix)')
plt.ylabel('Nhãn thực tế')
plt.xlabel('Nhãn dự đoán')
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()

# ======================
# 7. VISUALIZE DECISION TREE
# ======================
print("Đang xuất cây quyết định đầy đủ ra PDF...")

dot_data = export_graphviz(dt_model,
                           out_file=None,
                           feature_names=columns[:-1],
                           class_names=['Không TD', 'Có TD'],
                           filled=True,
                           rounded=True,
                           special_characters=True,
                           proportion=False,
                           precision=2)

# Chuyển thành PDF
graph = graphviz.Source(dot_data)
graph.render("decision_tree_COMPLETE", format="pdf", cleanup=True)
print("✅ Đã xuất file: decision_tree_COMPLETE.pdf")
print("   Mở file PDF này để xem toàn bộ cây có thể zoom thoải mái!")
# ======================
# 8. FEATURE IMPORTANCE
# ======================
feature_importance = pd.DataFrame({
    'feature': columns[:-1],
    'importance': dt_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "="*50)
print("ĐỘ QUAN TRỌNG CỦA CÁC ĐẶC TRƯNG")
print("="*50)
print(feature_importance)

# Vẽ biểu đồ
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance, palette='viridis')
plt.title('ĐỘ QUAN TRỌNG CỦA CÁC ĐẶC TRƯNG (Feature Importance)')
plt.xlabel('Mức độ quan trọng')
plt.ylabel('Đặc trưng')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.show()

# ======================
# 9. HYPERPARAMETER TUNING (OPTIONAL)
# ======================
print("\n" + "="*50)
print("TỐI ƯU THAM SỐ (TUỲ CHỌN)")
print("="*50)

# Thử nghiệm với các tham số khác nhau
max_depths = [3, 4, 5, 6, 7, None]
train_scores = []
test_scores = []

for depth in max_depths:
    dt_temp = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt_temp.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, dt_temp.predict(X_train)))
    test_scores.append(accuracy_score(y_test, dt_temp.predict(X_test)))

# Vẽ đồ thị
plt.figure(figsize=(10, 6))
plt.plot(max_depths[:-1], train_scores[:-1], 'o-', label='Train Score', linewidth=2)
plt.plot(max_depths[:-1], test_scores[:-1], 's-', label='Test Score', linewidth=2)
plt.xlabel('Độ sâu tối đa (Max Depth)')
plt.ylabel('Độ chính xác (Accuracy)')
plt.title('ẢNH HƯỞNG CỦA MAX_DEPTH ĐẾN HIỆU SUẤT')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('hyperparameter_tuning.png', dpi=300)
plt.show()

# ======================
# 10. SO SÁNH VỚI MÔ HÌNH ĐƠN GIẢN (BASELINE)
# ======================
from sklearn.dummy import DummyClassifier

# Baseline model: luôn dự đoán lớp đa số
baseline = DummyClassifier(strategy='most_frequent')
baseline.fit(X_train, y_train)
baseline_score = accuracy_score(y_test, baseline.predict(X_test))

print("\n" + "="*50)
print("SO SÁNH VỚI BASELINE MODEL")
print("="*50)
print(f"Baseline (luôn đoán 'Không TD'): {baseline_score:.4f}")
print(f"Decision Tree Model: {test_accuracy:.4f}")
print(f"Cải thiện: {(test_accuracy - baseline_score)*100:.2f}%")

# ======================
# 11. DEMO DỰ ĐOÁN MẪU MỚI
# ======================
print("\n" + "="*50)
print("DEMO DỰ ĐOÁN CHO BỆNH NHÂN MỚI")
print("="*50)

# Tạo dữ liệu mẫu (thay đổi giá trị để thử nghiệm)
sample_patient = np.array([[2, 120, 70, 25, 100, 25.5, 0.3, 35]])  # 1 mẫu
sample_scaled = scaler.transform(sample_patient)
prediction = dt_model.predict(sample_scaled)
pred_proba = dt_model.predict_proba(sample_scaled)

print(f"\nThông tin bệnh nhân mẫu:")
for i, col in enumerate(columns[:-1]):
    print(f"  {col}: {sample_patient[0][i]}")

print(f"\nKết quả dự đoán: {'CÓ TIỂU ĐƯỜNG' if prediction[0]==1 else 'KHÔNG TIỂU ĐƯỜNG'}")
print(f"Xác suất: Không TD: {pred_proba[0][0]:.2%}, Có TD: {pred_proba[0][1]:.2%}")

# ======================
# 12. LƯU MÔ HÌNH VÀ KẾT QUẢ
# ======================
import joblib
import json

# Lưu mô hình
joblib.dump(dt_model, 'decision_tree_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Lưu kết quả đánh giá
results = {
    'train_accuracy': float(train_accuracy),
    'test_accuracy': float(test_accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'best_params': {'max_depth': 4, 'criterion': 'gini'}
}

with open('evaluation_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("\n" + "="*50)
print("ĐÃ LƯU MÔ HÌNH VÀ KẾT QUẢ")
print("="*50)
print("✓ Mô hình: decision_tree_model.pkl")
print("✓ Scaler: scaler.pkl")
print("✓ Kết quả: evaluation_results.json")
print("✓ Hình ảnh: data_distribution.png, confusion_matrix.png,")
print("             decision_tree.png, feature_importance.png,")
print("             hyperparameter_tuning.png")


# ======================
# 13. DỰ ĐOÁN CHO NHIỀU BỆNH NHÂN MẪU
# ======================
print("\n" + "="*60)
print("DỰ ĐOÁN CHO 10 BỆNH NHÂN MẪU")
print("="*60)

# Tạo dữ liệu 10 bệnh nhân mẫu (5 có, 5 không + ngẫu nhiên)
sample_patients = np.array([
    # Không tiểu đường (dự kiến)
    [1, 85, 66, 29, 0, 26.6, 0.351, 31],    # Bệnh nhân 1: trẻ, glucose thấp
    [3, 89, 66, 23, 94, 28.1, 0.167, 21],   # Bệnh nhân 2: BMI bình thường
    [2, 100, 70, 27, 168, 23.9, 0.260, 22], # Bệnh nhân 3: glucose cao nhẹ
    [1, 95, 74, 25, 80, 25.9, 0.163, 24],   # Bệnh nhân 4: thông số bình thường
    [4, 110, 72, 35, 0, 26.2, 0.158, 30],   # Bệnh nhân 5: mang thai nhiều
    
    # Có tiểu đường (dự kiến)
    [8, 183, 64, 0, 0, 23.3, 0.672, 32],    # Bệnh nhân 6: glucose rất cao
    [10, 168, 74, 0, 0, 38.0, 0.537, 34],   # Bệnh nhân 7: BMI cao, glucose cao
    [7, 129, 86, 30, 180, 35.7, 0.916, 49], # Bệnh nhân 8: nhiều chỉ số cao
    [8, 180, 78, 32, 250, 43.3, 1.213, 52], # Bệnh nhân 9: tuổi cao, BMI rất cao
    [5, 148, 72, 35, 0, 33.6, 0.627, 50]    # Bệnh nhân 10: glucose cao + tuổi
])

# Chuẩn hóa dữ liệu mẫu
sample_scaled = scaler.transform(sample_patients)

# Dự đoán
predictions = dt_model.predict(sample_scaled)
prediction_probas = dt_model.predict_proba(sample_scaled)

# Tạo DataFrame để hiển thị đẹp
results_df = pd.DataFrame(sample_patients, columns=columns[:-1])

# Thêm cột kết quả
results_df['Dự đoán'] = ['CÓ TIỂU ĐƯỜNG' if p == 1 else 'KHÔNG TIỂU ĐƯỜNG' for p in predictions]
results_df['Xác suất Không TD'] = [f"{prob[0]:.1%}" for prob in prediction_probas]
results_df['Xác suất Có TD'] = [f"{prob[1]:.1%}" for prob in prediction_probas]
results_df['ID Bệnh nhân'] = [f"BN-{i+1:02d}" for i in range(len(sample_patients))]

# Sắp xếp lại cột
cols_order = ['ID Bệnh nhân'] + columns[:-1] + ['Dự đoán', 'Xác suất Không TD', 'Xác suất Có TD']
results_df = results_df[cols_order]

print("\nBẢNG DỰ ĐOÁN CHO 10 BỆNH NHÂN MẪU:")
print("-" * 120)
print(results_df.to_string(index=False))
print("-" * 120)

# ======================
# 14. PHÂN TÍCH CHI TIẾT CHO MỘT SỐ TRƯỜNG HỢP ĐẶC BIỆT
# ======================
print("\n" + "="*60)
print("PHÂN TÍCH CHI TIẾT CÁC TRƯỜNG HỢP ĐẶC BIỆT")
print("="*60)

# Hàm phân tích từng bệnh nhân
def analyze_patient(patient_id, patient_data, prediction, proba):
    print(f"\n🔍 PHÂN TÍCH BỆNH NHÂN {patient_id}:")
    print(f"   Kết quả: {prediction}")
    print(f"   Xác suất: Không TD: {proba[0]:.1%}, Có TD: {proba[1]:.1%}")
    
    # Đánh dấu các chỉ số nguy hiểm
    risk_factors = []
    
    if patient_data[1] > 140:  # Glucose
        risk_factors.append(f"Glucose cao ({patient_data[1]} > 140)")
    if patient_data[5] > 30:   # BMI
        risk_factors.append(f"BMI cao ({patient_data[5]:.1f} > 30)")
    if patient_data[7] > 45:   # Age
        risk_factors.append(f"Tuổi cao ({patient_data[7]} > 45)")
    if patient_data[0] > 6:    # Pregnancies
        risk_factors.append(f"Mang thai nhiều lần ({patient_data[0]})")
    
    if risk_factors:
        print(f"   ⚠️  YẾU TỐ NGUY CƠ: {', '.join(risk_factors)}")
    else:
        print(f"   ✅ KHÔNG có yếu tố nguy cơ rõ rệt")

# Phân tích 4 trường hợp điển hình
special_cases = [0, 5, 2, 8]  # BN-01, BN-06, BN-03, BN-09
for idx in special_cases:
    analyze_patient(
        results_df.iloc[idx]['ID Bệnh nhân'],
        sample_patients[idx],
        results_df.iloc[idx]['Dự đoán'],
        prediction_probas[idx]
    )

# ======================
# 15. VẼ BIỂU ĐỒ SO SÁNH BỆNH NHÂN
# ======================
plt.figure(figsize=(15, 10))

# Chọn 3 chỉ số quan trọng nhất để visualize
important_features = ['Glucose', 'BMI', 'Age']
feature_indices = [columns.index(f) for f in important_features]

# Tạo subplot
for i, (feature, idx) in enumerate(zip(important_features, feature_indices), 1):
    plt.subplot(2, 2, i)
    
    # Tách bệnh nhân có và không có tiểu đường
    diabetic_idx = [j for j, p in enumerate(predictions) if p == 1]
    non_diabetic_idx = [j for j, p in enumerate(predictions) if p == 0]
    
    # Giá trị của các bệnh nhân
    diabetic_values = sample_patients[diabetic_idx, idx]
    non_diabetic_values = sample_patients[non_diabetic_idx, idx]
    
    # Vẽ boxplot
    box_data = [non_diabetic_values, diabetic_values]
    box = plt.boxplot(box_data, labels=['Không TD', 'Có TD'], patch_artist=True)
    
    # Tô màu
    box['boxes'][0].set_facecolor('lightblue')
    box['boxes'][1].set_facecolor('lightcoral')
    
    # Đường ngưỡng nguy hiểm
    if feature == 'Glucose':
        plt.axhline(y=140, color='red', linestyle='--', alpha=0.5, label='Ngưỡng nguy hiểm (140)')
    elif feature == 'BMI':
        plt.axhline(y=30, color='red', linestyle='--', alpha=0.5, label='Ngưỡng béo phì (30)')
    
    plt.title(f'Phân bố {feature} theo nhóm')
    plt.ylabel(feature)
    plt.grid(True, alpha=0.3)
    if i == 1:
        plt.legend()

# Subplot thứ 4: Biểu đồ radar cho 1 bệnh nhân có và 1 không có
plt.subplot(2, 2, 4)

# Chọn 1 bệnh nhân mỗi loại
normal_patient = sample_patients[0]  # BN-01
diabetic_patient = sample_patients[5]  # BN-06

# Chuẩn hóa giá trị để vẽ radar chart
def normalize_for_radar(values):
    max_vals = sample_patients.max(axis=0)
    min_vals = sample_patients.min(axis=0)
    return [(v - min_vals[i]) / (max_vals[i] - min_vals[i]) for i, v in enumerate(values)]

norm_normal = normalize_for_radar(normal_patient[:5])  # Lấy 5 features đầu
norm_diabetic = normalize_for_radar(diabetic_patient[:5])

# Số lượng features
N = len(norm_normal)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # Đóng vòng

norm_normal += norm_normal[:1]
norm_diabetic += norm_diabetic[:1]

ax = plt.subplot(2, 2, 4, polar=True)
ax.plot(angles, norm_normal, 'o-', linewidth=2, label='BN-01 (Không TD)')
ax.fill(angles, norm_normal, alpha=0.25)
ax.plot(angles, norm_diabetic, 'o-', linewidth=2, label='BN-06 (Có TD)')
ax.fill(angles, norm_diabetic, alpha=0.25)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(columns[:5], fontsize=9)
ax.set_title('SO SÁNH HỒ SƠ BỆNH NHÂN', fontsize=12, y=1.1)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
plt.tight_layout()

plt.savefig('patient_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ======================
# 16. TẠO BẢNG TỔNG HỢP THỐNG KÊ
# ======================
print("\n" + "="*60)
print("THỐNG KÊ TỔNG HỢP 10 BỆNH NHÂN MẪU")
print("="*60)

# Tính thống kê
total_patients = len(sample_patients)
diabetic_count = sum(predictions == 1)
non_diabetic_count = sum(predictions == 0)

# Tính giá trị trung bình của các chỉ số
avg_diabetic = sample_patients[predictions == 1].mean(axis=0)
avg_non_diabetic = sample_patients[predictions == 0].mean(axis=0)

# Tạo bảng so sánh
comparison_df = pd.DataFrame({
    'Chỉ số': columns[:-1],
    'Trung bình (Không TD)': avg_non_diabetic,
    'Trung bình (Có TD)': avg_diabetic,
    'Chênh lệch': avg_diabetic - avg_non_diabetic
})

print(f"\n📊 TỔNG SỐ BỆNH NHÂN: {total_patients}")
print(f"   • Không tiểu đường: {non_diabetic_count} ({non_diabetic_count/total_patients:.0%})")
print(f"   • Có tiểu đường: {diabetic_count} ({diabetic_count/total_patients:.0%})")

print("\n📈 SO SÁNH GIÁ TRỊ TRUNG BÌNH:")
print(comparison_df.to_string(index=False, float_format=lambda x: f'{x:.2f}'))

# ======================
# 17. DỰ ĐOÁN TƯƠNG TÁC - CHO NGƯỜI DÙNG NHẬP LIỆU
# ======================
print("\n" + "="*60)
print("CHƯƠNG TRÌNH DỰ ĐOÁN TƯƠNG TÁC")
print("="*60)

def predict_interactive():
    print("\n🎯 NHẬP THÔNG TIN BỆNH NHÂN ĐỂ DỰ ĐOÁN:")
    print("   (Nhập 'q' để thoát)")
    print("-" * 40)
    
    while True:
        try:
            print("\n📝 NHẬP THÔNG TIN BỆNH NHÂN MỚI:")
            
            # Nhập từng giá trị
            values = []
            for i, feature in enumerate(columns[:-1]):
                while True:
                    try:
                        value = input(f"  {feature}: ")
                        if value.lower() == 'q':
                            print("👋 Kết thúc chương trình!")
                            return
                        values.append(float(value))
                        break
                    except ValueError:
                        print(f"    ⚠️ Vui lòng nhập số hợp lệ cho {feature}")
            
            # Chuyển đổi thành numpy array
            patient_data = np.array([values])
            
            # Chuẩn hóa và dự đoán
            patient_scaled = scaler.transform(patient_data)
            prediction = dt_model.predict(patient_scaled)[0]
            proba = dt_model.predict_proba(patient_scaled)[0]
            
            # Hiển thị kết quả
            print("\n" + "═" * 50)
            print("🔬 KẾT QUẢ DỰ ĐOÁN:")
            print("═" * 50)
            
            if prediction == 1:
                print(f"   ⚠️  CHẨN ĐOÁN: CÓ TIỂU ĐƯỜNG")
                print(f"   📊 XÁC SUẤT: {proba[1]:.1%}")
                print(f"\n   💡 KIẾN NGHỊ:")
                print(f"      • Kiểm tra HbA1c để xác nhận")
                print(f"      • Thay đổi chế độ ăn uống")
                print(f"      • Tập thể dục thường xuyên")
                print(f"      • Theo dõi đường huyết hàng ngày")
            else:
                print(f"   ✅ CHẨN ĐOÁN: KHÔNG TIỂU ĐƯỜNG")
                print(f"   📊 XÁC SUẤT: {proba[0]:.1%}")
                print(f"\n   💡 KIẾN NGHỊ:")
                print(f"      • Duy trì lối sống lành mạnh")
                print(f"      • Kiểm tra sức khỏe định kỳ")
                print(f"      • Giữ cân nặng hợp lý")
            
            # Đánh dấu các chỉ số nguy cơ
            print(f"\n   📋 CHỈ SỐ CẢNH BÁO:")
            for i, feature in enumerate(columns[:-1]):
                value = values[i]
                warning = ""
                
                if feature == 'Glucose' and value > 140:
                    warning = " (CAO - Nguy cơ cao)"
                elif feature == 'BMI' and value > 30:
                    warning = " (CAO - Béo phì)"
                elif feature == 'Age' and value > 50:
                    warning = " (CAO - Tuổi nguy cơ)"
                elif feature == 'BloodPressure' and value > 130:
                    warning = " (CAO - Cao huyết áp)"
                
                if warning:
                    print(f"      • {feature}: {value}{warning}")
            
            print("═" * 50)
            print("\n" + "─" * 40)
            
            # Hỏi có tiếp tục không
            cont = input("Tiếp tục dự đoán? (y/n): ")
            if cont.lower() != 'y':
                print("👋 Kết thúc chương trình!")
                break
                
        except Exception as e:
            print(f"Lỗi: {e}. Vui lòng thử lại!")

# Chạy chương trình tương tác (bỏ comment để dùng)
# predict_interactive()

# ======================
# 18. LƯU KẾT QUẢ DỰ ĐOÁN VÀO FILE
# ======================
print("\n" + "="*60)
print("LƯU KẾT QUẢ DỰ ĐOÁN VÀO FILE")
print("="*60)

# Tạo DataFrame kết quả chi tiết
detailed_results = []

for i in range(len(sample_patients)):
    detailed_results.append({
        'ID_BenhNhan': f'BN-{i+1:02d}',
        'Glucose': sample_patients[i][1],
        'BMI': sample_patients[i][5],
        'Age': sample_patients[i][7],
        'DuDoan': 'Co_Tieu_Duong' if predictions[i] == 1 else 'Khong_Tieu_Duong',
        'XacSuat_KhongTD': f"{prediction_probas[i][0]:.3f}",
        'XacSuat_CoTD': f"{prediction_probas[i][1]:.3f}",
        'MucDoTinCay': 'CAO' if max(prediction_probas[i]) > 0.8 else 'TRUNG_BINH' if max(prediction_probas[i]) > 0.6 else 'THAP'
    })

detailed_df = pd.DataFrame(detailed_results)

# Lưu ra file CSV
detailed_df.to_csv('ket_qua_du_doan.csv', index=False, encoding='utf-8-sig')

# Lưu ra file Excel với định dạng đẹp
with pd.ExcelWriter('ket_qua_du_doan.xlsx', engine='openpyxl') as writer:
    detailed_df.to_excel(writer, sheet_name='DuDoan', index=False)
    
    # Tạo sheet thống kê
    stats_df = pd.DataFrame({
        'ThongKe': ['TongSo', 'Co_Tieu_Duong', 'Khong_Tieu_Duong', 'TyLeCoTD', 'DoChinhXacTrungBinh'],
        'GiaTri': [total_patients, diabetic_count, non_diabetic_count, 
                  diabetic_count/total_patients, test_accuracy]
    })
    stats_df.to_excel(writer, sheet_name='ThongKe', index=False)

print("✅ Đã lưu kết quả dự đoán:")
print("   • ket_qua_du_doan.csv")
print("   • ket_qua_du_doan.xlsx")
print("\n📊 KẾT QUẢ TÓM TẮT:")
print(f"   • Tổng bệnh nhân: {total_patients}")
print(f"   • Dự đoán CÓ tiểu đường: {diabetic_count} ({diabetic_count/total_patients:.0%})")
print(f"   • Dự đoán KHÔNG tiểu đường: {non_diabetic_count} ({non_diabetic_count/total_patients:.0%})")
print(f"   • Độ chính xác mô hình: {test_accuracy:.1%}")

# ======================
# 19. TẠO BÁO CÁO TỰ ĐỘNG (AUTOMATIC REPORT)
# ======================
report_content = f"""
BÁO CÁO KẾT QUẢ DỰ ĐOÁN TIỂU ĐƯỜNG
{'='*60}

I. THÔNG TIN MÔ HÌNH
- Thuật toán: Cây Quyết Định (Decision Tree)
- Ngày chạy: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}
- Độ chính xác trên tập test: {test_accuracy:.2%}

II. KẾT QUẢ DỰ ĐOÁN CHO {total_patients} BỆNH NHÂN MẪU
- Số bệnh nhân dự đoán CÓ tiểu đường: {diabetic_count} ({diabetic_count/total_patients:.0%})
- Số bệnh nhân dự đoán KHÔNG tiểu đường: {non_diabetic_count} ({non_diabetic_count/total_patients:.0%})

III. PHÂN TÍCH ĐẶC ĐIỂM NHÓM
A. Nhóm CÓ tiểu đường (trung bình):
   - Glucose: {avg_diabetic[1]:.1f} mg/dL
   - BMI: {avg_diabetic[5]:.1f}
   - Tuổi: {avg_diabetic[7]:.1f} tuổi

B. Nhóm KHÔNG tiểu đường (trung bình):
   - Glucose: {avg_non_diabetic[1]:.1f} mg/dL
   - BMI: {avg_non_diabetic[5]:.1f}
   - Tuổi: {avg_non_diabetic[7]:.1f} tuổi

IV. KHUYẾN NGHỊ
1. Bệnh nhân có chỉ số Glucose > 140 cần được kiểm tra thêm
2. BMI > 30 là yếu tố nguy cơ quan trọng
3. Tuổi > 45 làm tăng nguy cơ mắc bệnh

{'='*60}
Mô hình này chỉ mang tính chất tham khảo, không thay thế chẩn đoán của bác sĩ.
"""

# Lưu báo cáo
with open('bao_cao_ket_qua.txt', 'w', encoding='utf-8') as f:
    f.write(report_content)

print("\n📄 Đã tạo báo cáo tự động: bao_cao_ket_qua.txt")