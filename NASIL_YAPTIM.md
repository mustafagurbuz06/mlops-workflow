# Banka Kampanyası Modelini Nasıl Geliştirdim

Bir banka, müşteriyi aramadan önce vadeli mevduata abone olup olmayacağını tahmin etmek istiyor. Veri setinde Mayıs 2008 – Kasım 2010 arasındaki yaklaşık 41.000 telefon görüşmesi var, satırlar zaman sırasına göre dizili.

Dersteki notebook'ta (`lesson.ipynb`) sırasıyla şunlar vardı:

1. `duration` (görüşme süresi) sütunu çıkarıldı, çünkü görüşme bitmeden bilinmiyor.
2. Veri dosya sırasına göre bölündü: ilk %70 eğitim, sonraki %15 test, son %15 "live" dönemi.
3. Sayısal sütunlar ölçeklendi, kategorik sütunlar 0/1 sütunlarına çevrildi (One-Hot).
4. `class_weight="balanced"` ile Logistic Regression eğitildi.
5. Model Docker ile servis haline getirildi ve live dönemi için drift (PSI) ölçüldü.

Bu modelin test sonucu **ROC-AUC = 0.661** idi. Accuracy ise sadece **0.25** çıkıyordu.

## Fark ettiğim sorunlar

**Model neredeyse herkese "abone olur" diyordu.** Test döneminde gerçekte müşterilerin %11'i abone olmuş, ama model %85'ine "yes" diyor. Accuracy'nin 0.25 çıkmasının sebebi bu.

**Bunun sebebi makro ekonomik sütunlar.** `euribor3m` (faiz), `nr.employed` (istihdam), `emp.var.rate` gibi beş sütun müşteriyle değil, o günün ekonomisiyle ilgili. 2008 krizinden sonra bu değerler eğitim verisinde hiç görülmemiş seviyelere düştü. Mesela faiz eğitimde 1.41 ile 5.04 arasında, testte ise 1.25 ile 1.41 arasında. Model eğitimde "faiz düştükçe abonelik artıyor" öğrenmiş. Testte faiz eğitimdeki en düşük değerin de altına inince bu kuralı aşırı uyguluyor ve herkese yüksek olasılık veriyor. Orijinal notebook'un drift tablosunda en çok kayan beş sütun da tam olarak bunlar.

**Eğitim verisinin çoğu bugünü temsil etmiyor.** Eğitim verisinin büyük kısmı 2008 yazından; o dönemde abone olma oranı %3–6 ve "önceki kampanyanın sonucu" bilgisi hiç yok. Test dönemi (2009 baharı) bambaşka bir dönem. En çok benzeyen veri, eğitim döneminin son ayları.

**Model bir kez eğitilip hiç güncellenmiyor.** Veri sürekli değişirken model 2008'de öğrendikleriyle kalıyor.

## 1. aşama: Daha iyi bir model

Modeli değiştirmedim, yine Logistic Regression. Değiştirdiğim şey, modelin hangi veriyi nasıl kullandığı:

1. **Makro ekonomik sütunları çıkardım.** Dönem değiştiğinde en çok kayan ve modeli yanıltan sütunlar bunlar.
2. **Yakın geçmişe daha fazla ağırlık verdim.** Her satıra bir ağırlık verdim: En yeni görüşme 1, 2000 satır önceki görüşme 0.5, 4000 satır önceki 0.25 ağırlık alıyor. Böylece model test dönemine en çok benzeyen son aylardan daha çok öğreniyor.
3. **Regularization'ı güçlendirdim (`C=0.01`).** Ağırlık verince etkili örnek sayısı azalıyor; güçlü regularization modelin bu az sayıdaki örneğe aşırı uymasını engelliyor.
4. **Yaşı eğrisel modelledim (spline).** Abone olma oranı yaşla düz bir çizgide artmıyor: Öğrencilerde (gençler) ve emeklilerde (yaşlılar) yüksek, ortada düşük. Düz bir çizgi bunu yakalayamıyor.
5. **Karar eşiğini ayarladım.** Orijinal model 0.5 eşiğini kullanıyordu. Eşiği, test döneminde F1 skorunu en yüksek yapan değere (0.60) çektim ve servisin de bu eşiği kullanmasını sağladım.

Her adımın katkısı (test dönemi):

| Adım | ROC-AUC | PR-AUC |
|---|---|---|
| Orijinal model | 0.661 | 0.190 |
| Makro sütunlar çıkarıldı | 0.661 | 0.199 |
| + yakın geçmiş ağırlığı | 0.671 | 0.215 |
| + güçlü regularization | 0.690 | 0.251 |
| + yaş spline | **0.695** | **0.257** |

Ağırlık ve regularization ayarlarını seçerken birkaç değer denedim. Skorlar geniş bir bölgede birbirine yakın çıktı (0.687–0.703). En yüksek değeri değil, bu bölgenin ortasına yakın bir değeri seçtim; böylece teste aşırı uydurma riskini azalttım.

Bunların dışında yeni özellikler (eğitim × meslek, iletişim × ay gibi), sütunlar arası etkileşimler ve gradient boosting modelleri de denedim. Hiçbiri skoru artırmadı, bu yüzden modeli sade tuttum.

## 2. aşama: Sürekli eğitim (continuous training)

Müşterinin abone olup olmadığı görüşmenin sonunda belli oluyor. Yani banka, modeli düzenli aralıklarla o ana kadar biriken verilerle yeniden eğitebilir. 250 görüşme, test döneminde yaklaşık 1 güne, aramaların seyrekleştiği live döneminde ise yaklaşık 2 haftaya denk geliyor. Bunu simüle ettim: Test dönemini 250 görüşmelik parçalara böldüm ve her parçayı, **sadece o parçadan önceki görüşmelerle** eğitilmiş yeni bir modelle tahmin ettim. Hiçbir tahmin, o görüşme anında bilinmeyen bir bilgiyi kullanmıyor.

Burada makro sütunları geri ekledim. Makro sütunların sorunu, bir kez eğitilen modelin eski ekonomik değerlere takılı kalmasıydı. Model düzenli yeniden eğitilince bu sütunlar hep güncel ekonomiyi yansıtıyor ve faydalı hale geliyor. Test döneminde iki seçeneği de denedim, makrolu versiyon biraz daha iyi çıktı.

Adil olsun diye orijinal modele de aynı imkânı tanıdım.

## Sonuçlar (test dönemi)

| | Orijinal | Son model (bir kez eğitilir) | Son model + sürekli eğitim |
|---|---|---|---|
| **ROC-AUC** | 0.661 | 0.695 | **0.708** |
| PR-AUC | 0.190 | 0.257 | **0.259** |
| F1 (yes), ayarlı eşik | 0.211 | 0.313 | **0.338** |
| Accuracy, ayarlı eşik | 0.249 | **0.848** | 0.811 |
| "Yes" denen müşteri oranı | %85 | %12 | %18 |

Orijinal model de yeniden eğitilince test döneminde sadece 0.667'ye çıkıyor. Yani testteki kazancın büyük kısmı modelin kendisinden geliyor.

Orijinal model herkese "yes" diyerek abonelerin çoğunu yakalıyordu, ama aradığı her 100 kişiden sadece 12'si abone oluyordu. Son model daha seçici: Aradığı her 100 kişiden 27–30'u abone oluyor.

## Kendi sonucumu nasıl kontrol ettim

Model ayarlarını ve eşiği test dönemine bakarak seçtim. Yani test dönemi benim için bir validation görevi gördü ve oradaki skor biraz iyimser olabilir. Bunu kontrol etmek için **live dönemini** (Mayıs 2009 sonu – Kasım 2010) ayrı tuttum ve seçimleri hep test skoruna göre yaptım. Açık olmak gerekirse, geliştirme sırasında birkaç finalist modelin live skoruna da baktım; bu yüzden live sonuçları da çok az iyimser olabilir. Ama hiçbir ayarı live skorunu yükseltmek için seçmedim:

| Live dönemi | ROC-AUC | Accuracy | F1 (yes) |
|---|---|---|---|
| Orijinal | 0.563 | 0.431 | 0.540 |
| Son model (bir kez eğitilir) | 0.595 | 0.628 | 0.386 |
| Orijinal + sürekli eğitim | 0.693 | 0.437 | 0.571 |
| **Son model + sürekli eğitim** | **0.678** | **0.578** | **0.576** |

Bu tablodan çıkardığım sonuçlar:

- Hiç görülmemiş bu dönemde de modelim orijinalden iyi. Yani iyileşme sadece teste uydurmaktan kaynaklanmıyor.
- Live dönemi 17 ayı kapsıyor ve aylık abone olma oranı %14 ile %57 arasında değişiyor (test döneminde %11'di). Bu kadar uzun ve değişken bir dönemde **en büyük fark sürekli eğitimden geliyor**. Orijinal model bile yeniden eğitilince 0.563'ten 0.693'e çıkıyor; ROC-AUC'de benim yeniden eğitilen modelimin biraz önünde. Ama orijinal model orada da müşterilerin %93'üne "yes" diyor; benim model daha doğru karar veriyor (accuracy 0.58'e karşı 0.44).
- Yeniden eğitim olmadan, test dönemine göre seçtiğim eşik live döneminde çok az kişiye "yes" diyor ve F1 düşüyor. Bu, dersin konusu olan drift'in doğrudan bir sonucu.

Kısacası, bu veride uzun vadede en önemli şey **modeli güncel tutmak**. Bu da MLOps'un çözmeye çalıştığı problemin ta kendisi.

## Kodu nasıl düzenledim

- Bütün importları tek hücrede, ayarları (dosya yolu, split oranları, çıkarılan sütunlar, ağırlık, regularization ve yeniden eğitim sıklığı) tek hücrede topladım.
- Tekrar eden işleri fonksiyonlara taşıdım: veri yükleme, zamana göre bölme, model kurma, sürekli eğitim simülasyonu (`walk_forward`), değerlendirme, eşik seçimi.
- Orijinal notebook'ta sayısal ve kategorik sütunlar için iki kez yazılan PSI hesabını tek bir `psi()` fonksiyonuna indirdim. Sonuçlar orijinaliyle birebir aynı çıkıyor.
- Servis (`serve/app.py`) yine bütün sütunları içeren satırı kabul ediyor; model ihtiyaç duymadığı sütunları kendisi atıyor. Değişen tek şey, sabit 0.5 yerine ayarlanan eşiği kullanması.

## Çalıştırmak için

Bütün adımlar çıktılarıyla birlikte `improved_lesson.ipynb` dosyasında. `requirements.txt`'teki kütüphaneler dışında ek bir şey gerekmiyor. Docker adımları `lesson.ipynb` ile aynı.
