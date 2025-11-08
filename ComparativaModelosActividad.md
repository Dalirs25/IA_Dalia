EigenFace / FisherFace / LBPH vs MediaPipe Face Mesh para la Detección de emociones
Métodos

•	EigenFace (PCA): representa caras por componentes principales. Simple, pero sensible a iluminación y expresiones sutiles.

•	FisherFace (LDA): mejora la separación entre clases respecto a PCA, pero sigue limitado por iluminación y pose.

•	LBPH: descriptores locales de textura; más robusto a iluminación y útil en tiempo real, pero no modela geometría facial.

•	MediaPipe Face Mesh: malla de 468 landmarks 3D; permite medir geometría y movimientos (cejas, ojos, boca) en tiempo real — ideal para detectar emociones.

Comparativa
•	Robustez a iluminación: LBPH y MediaPipe tienen un rendimiento similar  y ests a su vez son mejores que Eigen/Fisher.

•	Información geométrica (movimientos faciales): MediaPipe es mejor que LBPH/Eigen/Fisher.

•	Detección de micro-movimientos / dinámica: MediaPipe es mejor porque permite usar derivadas temporales.

•	Facilidad para emociones : MediaPipe > LBPH > Fisher > Eigen.

Puntos de MediaPipe a usar (regiones clave)
•	Cejas (izq/der): elevación y fruncido.

•	Ojos (párpados superior/inferior): apertura, parpadeo.

•	Comisuras de la boca y centro de labios: sonrisa, asimetría, apertura.

•	Punta/ base de la nariz y aletas nasales: arrugamiento (asco).

•	Mentón/jaw: descenso o tensión (sorpresa/tristeza).

•	Contorno lateral de cara (para normalizar tamaño).

Normalización
Usar IOD o face_height como referencia: d_norm = d / D_ref. Así las medidas son proporcionales al tamaño de la cabeza.

Pasos
1.	Extraer landmarks con MediaPipe por frame y descartar frames con baja confianza.
2.	Calcular features normalizadas.
3.	Clasificar con un modelo simple (SVM/MLP) o LSTM para secuencias y suavizar salidas.
