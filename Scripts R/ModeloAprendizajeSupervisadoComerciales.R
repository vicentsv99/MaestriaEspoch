# Datos de entrada
dias <- c(1, 2, 3, 4, 5)
generacion_residuos <- c(3.569756098,2.135443038,1.696794872,2.643382353,3.1425)

# Crear un data frame con los datos
data <- data.frame(dias = dias, generacion_residuos = generacion_residuos)

# Crear el modelo de regresión lineal
modelo <- lm(generacion_residuos ~ dias, data = data)

# metricas del modelo
summary(modelo)
#R cuadrado y R cuadrado ajustado indican que el modelo explica aproximadamente la mitad de la variabilidad en la generación de residuos.

# Generar datos de entrada hasta el día 30
dias_prediccion <- 1:30
data_prediccion <- data.frame(dias = dias_prediccion)

# Predecir la generación de residuos para los días hasta el 30
predicciones <- predict(modelo, newdata = data_prediccion)

# Crear un data frame con los resultados de la predicción
resultados_prediccion <- data.frame(dias = dias_prediccion, generacion_residuos = predicciones)

# Crear el gráfico
library(ggplot2)

ggplot() +
  geom_point(data = data, aes(x = dias, y = generacion_residuos), color = "blue") +  # Puntos de datos originales
  geom_line(data = resultados_prediccion, aes(x = dias, y = generacion_residuos), color = "red") +  # Línea de predicciones del modelo
  labs(x = "Día", y = "Generación de Residuos", title = "Modelo de Regresión Lineal con Predicciones hasta el día 30") +
  theme_minimal()

# Predecir la generación de residuos para el día 10
dia_Pred <- data.frame(dias = 30)
prediccion_dia <- predict(modelo, newdata = dia_Pred)
cat("Predicción de generación de residuos para el día 10:", prediccion_dia, "\n")


# Calcular métricas

# Mean Squared Error (MSE)
mse <- mean((data$generacion_residuos - predicciones)^2)

# Root Mean Squared Error (RMSE)
rmse <- sqrt(mse)

# Mean Absolute Error (MAE)
mae <- mean(abs(data$generacion_residuos - predicciones))

# R-squared (R²)
rsq <- summary(modelo)$r.squared

# R-squared ajustado
rsq_adj <- summary(modelo)$adj.r.squared

cat("Mean Squared Error (MSE):", mse, "\n")
cat("Root Mean Squared Error (RMSE):", rmse, "\n")
cat("Mean Absolute Error (MAE):", mae, "\n")
cat("R-squared (R²):", rsq, "\n")
cat("R-squared ajustado:", rsq_adj, "\n")