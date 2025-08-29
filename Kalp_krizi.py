import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class HeartDiseasePredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("Kalp Hastalığı Tahmin Sistemi")
        self.root.geometry("500x500")
        self.root.configure(bg="#f0f0f0")

        # Model ve veri ön işleme nesnelerini tanımla
        self.model = None
        self.scaler = None
        self.label_encoder_sex = None
        self.label_encoder_cp = None

        # Modeli yükle
        self.load_and_train_model()

        # Arayüzü oluştur
        self.create_widgets()

    def show_model_comparison(self):
        comparison_text = """Model Karşılaştırma Matrisi:

 Model            Doğruluk        Hız         Yorumlama  
├─────────────────┼────────────┼────────────┼───────
 Yapay Sinir Ağı     %88.77        Orta         Zor        
├─────────────────┼────────────┼────────────┼───────
 Random Forest    %70.65         Hızlı        Kolay      
├─────────────────┼────────────┼────────────┼───────
 Lojistik Reg.         %85.51      Çok Hızlı    Çok Kolay  
├─────────────────┼────────────┼────────────┼───────
 Karar Ağaçları       %31.16        Hızlı          Kolay      
└─────────────────┴────────────┴────────────┴───────

Açıklamalar:
- Doğruluk: Modelin doğru tahmin yapma yüzdesi
- Hız: Tahmin yapma süresi
- Yorumlama: Sonuçların anlaşılabilirlik seviyesi

Not: Bu uygulamada Yapay Sinir Ağı modeli kullanılmaktadır."""

        messagebox.showinfo("Model Karşılaştırma", comparison_text)

    def show_normal_values(self):
        info_text = """Normal Değer Aralıkları:

Dinlenme Kan Basıncı (mmHg):
- Normal: 120/80 ve altı
- Yüksek Normal: 120-129/80
- Hipertansiyon: 130/80 ve üzeri

Kolesterol (mg/dL):
- Toplam Kolesterol:
  * Normal: 200'ün altı
  * Sınırda Yüksek: 200-239
  * Yüksek: 240 ve üzeri

Not: Bu değerler genel sağlıklı bireyler için referans değerlerdir. 
Kişisel sağlık durumunuz için doktorunuza danışınız."""

        messagebox.showinfo("Normal Değer Aralıkları", info_text)

    def show_chest_pain_info(self):
        chest_pain_text = """Göğüs Ağrısı Tipleri ve Risk Durumları:

ATA (Atypical Angina) – Atipik Anjina
Tanım: Kalp kaynaklı olabilir, ancak tipik belirtiler göstermez.
Risk Durumu: Orta riskli 

NAP (Non-Anginal Pain) – Anjina Olmayan Ağrı
Tanım: Kalp dışı nedenlere bağlı göğüs ağrısı.
Risk Durumu: Düşük riskli

TA (Typical Angina) – Tipik Anjina
Tanım: Klasik kalp kökenli göğüs ağrısı.
Risk Durumu: En az riskli 

ASY (Asymptomatic) – Asemptomatik
Tanım: Hiçbir belirti göstermeyen kişiler.
Risk Durumu: Yüksek riskli (belirtiler olmadığı için hastalık sessiz ilerler)."""

        messagebox.showinfo("Göğüs Ağrısı Tipleri", chest_pain_text)

    def load_and_train_model(self):
        # Veriyi yükle
        df = pd.read_csv("heart.csv")

        # Label encoder'ları oluştur
        label_encoder = LabelEncoder()
        self.label_encoder_sex = LabelEncoder()
        self.label_encoder_cp = LabelEncoder()

        # Kategorik verileri dönüştür
        df["Sex"] = self.label_encoder_sex.fit_transform(df["Sex"])
        df["ChestPainType"] = self.label_encoder_cp.fit_transform(df["ChestPainType"])
        df["RestingECG"] = label_encoder.fit_transform(df["RestingECG"])
        df["ExerciseAngina"] = label_encoder.fit_transform(df["ExerciseAngina"])
        df["ST_Slope"] = label_encoder.fit_transform(df["ST_Slope"])

        # Veriyi hazırla
        X = df.drop("HeartDisease", axis=1)
        y = df["HeartDisease"]

        # Veriyi böl
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # SMOTE uygula
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

        # Ölçeklendir
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_smote)
        X_test_scaled = self.scaler.transform(X_test)

        # Modeli oluştur ve eğit
        self.model = Sequential()
        self.model.add(Dense(16, input_dim=X_train_scaled.shape[1], activation="relu"))
        self.model.add(Dense(1, activation="sigmoid"))

        optimizer = Adam(learning_rate=0.001)
        self.model.compile(
            loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )

        self.model.fit(
            X_train_scaled,
            y_train_smote,
            epochs=100,
            verbose=0,
            validation_data=(X_test_scaled, y_test),
        )

    def create_widgets(self):
        # Ana frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Başlık ve butonlar için frame
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 10))

        # Başlık
        title_label = ttk.Label(
            title_frame,
            text="Kalp Hastalığı Tahmin Sistemi",
            font=("Arial", 20, "bold"),
            anchor="center",
        )
        title_label.pack(fill=tk.X, pady=10)

        # Butonlar için frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))

        # Model karşılaştırma butonu
        model_button = ttk.Button(
            button_frame,
            text="📊 Model Karşılaştırma",
            command=self.show_model_comparison,
        )
        model_button.pack(fill=tk.X, pady=2)

        # Bilgi butonu
        info_button = ttk.Button(
            button_frame, text="ℹ️ Normal Değerler", command=self.show_normal_values
        )
        info_button.pack(fill=tk.X, pady=2)

        # Göğüs ağrısı bilgi butonu
        chest_pain_button = ttk.Button(
            button_frame,
            text="💓 Göğüs Ağrısı Tipleri",
            command=self.show_chest_pain_info,
        )
        chest_pain_button.pack(fill=tk.X, pady=2)

        # Giriş alanları için frame
        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Yaş
        ttk.Label(input_frame, text="👤 Yaş:").grid(
            row=0, column=0, padx=5, pady=5, sticky="w"
        )
        self.age_entry = ttk.Entry(input_frame)
        self.age_entry.grid(row=0, column=1, padx=5, pady=5)

        # Cinsiyet
        ttk.Label(input_frame, text="🚻 Cinsiyet:").grid(
            row=1, column=0, padx=5, pady=5, sticky="w"
        )
        self.sex_var = tk.StringVar()
        self.sex_combo = ttk.Combobox(
            input_frame,
            textvariable=self.sex_var,
            values=["Erkek", "Kadın"],
            state="readonly",
        )
        self.sex_combo.grid(row=1, column=1, padx=5, pady=5)

        # Göğüs Ağrısı Tipi
        ttk.Label(input_frame, text="💢 Göğüs Ağrısı Tipi:").grid(
            row=2, column=0, padx=5, pady=5, sticky="w"
        )
        self.chest_pain_var = tk.StringVar()
        self.chest_pain_combo = ttk.Combobox(
            input_frame,
            textvariable=self.chest_pain_var,
            values=["ATA", "NAP", "ASY", "TA"],
            state="readonly",
        )
        self.chest_pain_combo.grid(row=2, column=1, padx=5, pady=5)

        # Dinlenme Kan Basıncı
        ttk.Label(input_frame, text="🩸 Dinlenme Kan Basıncı:").grid(
            row=3, column=0, padx=5, pady=5, sticky="w"
        )
        self.resting_bp_entry = ttk.Entry(input_frame)
        self.resting_bp_entry.grid(row=3, column=1, padx=5, pady=5)

        # Kolesterol
        ttk.Label(input_frame, text="🧪 Kolesterol:").grid(
            row=4, column=0, padx=5, pady=5, sticky="w"
        )
        self.cholesterol_entry = ttk.Entry(input_frame)
        self.cholesterol_entry.grid(row=4, column=1, padx=5, pady=5)

        # Tahmin butonu
        predict_button = ttk.Button(
            main_frame, text="🔮 Tahmin Et", command=self.predict
        )
        predict_button.pack(pady=20)

        # Sonuç etiketi
        self.result_label = ttk.Label(main_frame, text="", font=("Arial", 12))
        self.result_label.pack(pady=10)

    def predict(self):
        try:
            # Kullanıcı verilerini al
            age = float(self.age_entry.get())
            sex = "M" if self.sex_var.get() == "Erkek" else "F"
            chest_pain = self.chest_pain_var.get()
            resting_bp = float(self.resting_bp_entry.get())
            cholesterol = float(self.cholesterol_entry.get())

            # Veriyi hazırla
            user_data = pd.DataFrame(
                {
                    "Age": [age],
                    "Sex": [self.label_encoder_sex.transform([sex])[0]],
                    "ChestPainType": [self.label_encoder_cp.transform([chest_pain])[0]],
                    "RestingBP": [resting_bp],
                    "Cholesterol": [cholesterol],
                    "FastingBS": [0],
                    "RestingECG": [0],
                    "MaxHR": [150],
                    "ExerciseAngina": [0],
                    "Oldpeak": [0.0],
                    "ST_Slope": [1],
                }
            )

            # Veriyi ölçeklendir
            user_data_scaled = self.scaler.transform(user_data)

            # Tahmin yap
            prediction = self.model.predict(user_data_scaled)[0][0]

            # Sonucu göster
            risk_percentage = prediction * 100

            # Risk seviyesine göre öneriler
            if prediction > 0.5:
                risk_level = "Yüksek Risk"
                color = "red"

            else:
                risk_level = "Düşük Risk"
                color = "green"

            result_text = f"""
🔍 Tahmin Sonucu:

Risk Seviyesi: {risk_level}
Risk Oranı: {risk_percentage:.2f}%



Not: Bu sonuçlar tahmini değerlerdir. 
Kesin teşhis için doktorunuza danışınız."""

            self.result_label.config(
                text=result_text, foreground=color, font=("Arial", 11), justify="left"
            )

        except ValueError as e:
            messagebox.showerror("Hata", "Lütfen tüm alanları doğru formatta doldurun!")
        except Exception as e:
            messagebox.showerror("Hata", f"Bir hata oluştu: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = HeartDiseasePredictor(root)
    root.mainloop()
