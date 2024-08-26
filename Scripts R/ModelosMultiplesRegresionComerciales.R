# Cargar el paquete randomForest si no está cargado
if (!requireNamespace("randomForest", quietly = TRUE)) {
  install.packages("randomForest")
}
if (!requireNamespace("ggplot2", quietly = TRUE)) {
  install.packages("ggplot2")
}
if (!requireNamespace("xgboost", quietly = TRUE)) {
  install.packages("xgboost")
}
if (!requireNamespace("e1071", quietly = TRUE)) {
  install.packages("e1071")
}

library(e1071)
library(xgboost)
library(ggplot2)
library(randomForest)


# Datos originales
dias <- c(1, 2, 3, 4, 5)
generacion_residuos <- c(3.569756098,2.135443038,1.696794872,2.643382353,3.1425)

# Crear un data frame con los datos originales
data <- data.frame(dias = dias, generacion_residuos = generacion_residuos)

#########################################################################################
#         RANDOM FOREST #
#########################################################################################

# Entrenar el modelo Random Forest
modelo_rf <- randomForest(generacion_residuos ~ dias, data = data, ntree = 100)
predicciones_rf <- predict(modelo_rf, data)

# Calcular las métricas de evaluación

# Mean Squared Error (MSE)
mse_rf <- mean((data$generacion_residuos - predicciones_rf)^2)
# Root Mean Squared Error (RMSE)
rmse_rf <- sqrt(mse_rf)
# Mean Absolute Error (MAE)
mae_rf <- mean(abs(data$generacion_residuos - predicciones_rf))
# R-squared (R²)
rsq_rf <- cor(data$generacion_residuos, predicciones_rf)^2
cat("Mean Squared Error (MSE):", mse_rf, "\n")
cat("Root Mean Squared Error (RMSE):", rmse_rf, "\n")
cat("Mean Absolute Error (MAE):", mae_rf, "\n")
cat("R-squared (R²):", rsq_rf, "\n")


# Generar datos de predicción para visualización
dias_pred <- seq(1, 30, by = 1)  # Generar secuencia de días hasta el día 30
data_pred <- data.frame(dias = dias_pred)

predicciones_rf130 <- predict(modelo_rf, newdata = data_pred)

# Crear data frame para la visualización
df_predicciones_rf <- data.frame(dias = dias_pred, generacion_residuos = predicciones_rf130)
df_predicciones_rf
# Graficar datos originales y predicciones del modelo


ggplot() +
  geom_point(data = data, aes(x = dias, y = generacion_residuos), color = "blue") +  # Puntos de datos originales
  geom_line(data = df_predicciones_rf, aes(x = dias, y = generacion_residuos), color = "red") +  # Línea de predicciones del modelo
  labs(x = "Día", y = "Generación de Residuos", title = "Modelo de Random Forest para Generación de Residuos") +
  theme_minimal()

#########################################################################################
#         Modelo XGBoost #
#########################################################################################

# Preparar los datos para XGBoost
data_matrix <- model.matrix(generacion_residuos ~ dias - 1, data = data)
dtrain <- xgb.DMatrix(data = data_matrix, label = data$generacion_residuos)

# Ajustar un modelo de XGBoost
param <- list(objective = "reg:squarederror", max_depth = 3, eta = 0.1)
modelo_xgb <- xgb.train(params = param, data = dtrain, nrounds = 100)

# Predicciones del modelo en los datos originales
predicciones_xgb <- predict(modelo_xgb, data_matrix)

# Calcular las métricas de evaluación para el modelo de XGBoost

# Mean Squared Error (MSE)
mse_xgb <- mean((data$generacion_residuos - predicciones_xgb)^2)
# Root Mean Squared Error (RMSE)
rmse_xgb <- sqrt(mse_xgb)
# Mean Absolute Error (MAE)
mae_xgb <- mean(abs(data$generacion_residuos - predicciones_xgb))
# R-squared (R²)
rsq_xgb <- cor(data$generacion_residuos, predicciones_xgb)^2
cat("Mean Squared Error (MSE):", mse_xgb, "\n")
cat("Root Mean Squared Error (RMSE):", rmse_xgb, "\n")
cat("Mean Absolute Error (MAE):", mae_xgb, "\n")
cat("R-squared (R²):", rsq_xgb, "\n")


# Generar datos de predicción para visualización
data_pred <- data.frame(dias = seq(1, 30, by = 1))
data_pred_matrix <- model.matrix(~ dias - 1, data = data_pred)
predicciones <- predict(modelo_xgb, newdata = data_pred_matrix)

# Crear data frame para la visualización
df_predicciones <- data.frame(dias = data_pred$dias, generacion_residuos = predicciones)

# Graficar datos originales y predicciones del modelo
ggplot() +
  geom_point(data = data, aes(x = dias, y = generacion_residuos), color = "blue", size = 3) +  # Puntos de datos originales
  geom_line(data = df_predicciones, aes(x = dias, y = generacion_residuos), color = "red", size = 1) +  # Línea de predicciones del modelo
  labs(x = "Día", y = "Generación de Residuos", title = "Modelo de Gradient Boosting (XGBoost) para Generación de Residuos") +
  theme_minimal()

#########################################################################################
#        Modelo de Regresión con Soporte Vectorial (SVM) #
#########################################################################################

# Entrenar el modelo de SVM
modelo_svm <- svm(generacion_residuos ~ dias, data = data, kernel = "radial")
predicciones_svm <- predict(modelo_svm, data)

# Calcular las métricas de evaluación para el modelo SVM

# Mean Squared Error (MSE)
mse_svm <- mean((data$generacion_residuos - predicciones_svm)^2)
# Root Mean Squared Error (RMSE)
rmse_svm <- sqrt(mse_svm)
# Mean Absolute Error (MAE)
mae_svm <- mean(abs(data$generacion_residuos - predicciones_svm))
# R-squared (R²)
rsq_svm <- cor(data$generacion_residuos, predicciones_svm)^2
cat("Mean Squared Error (MSE):", mse_svm, "\n")
cat("Root Mean Squared Error (RMSE):", rmse_svm, "\n")
cat("Mean Absolute Error (MAE):", mae_svm, "\n")
cat("R-squared (R²):", rsq_svm, "\n")

# Generar datos de predicción para visualización
dias_pred <- seq(1, 30, by = 1)  # Generar secuencia de días hasta el día 30
data_pred <- data.frame(dias = dias_pred)
predicciones <- predict(modelo_svm, newdata = data_pred)

# Crear data frame para la visualización
df_predicciones <- data.frame(dias = dias_pred, generacion_residuos = predicciones)

# Graficar datos originales y predicciones del modelo
ggplot() +
  geom_point(data = data, aes(x = dias, y = generacion_residuos), color = "blue", size = 3) +  # Puntos de datos originales
  geom_line(data = df_predicciones, aes(x = dias, y = generacion_residuos), color = "red", size = 1) +  # Línea de predicciones del modelo
  labs(x = "Día", y = "Generación de Residuos", title = "Modelo de Regresión con SVM para Generación de Residuos") +
  theme_minimal()
