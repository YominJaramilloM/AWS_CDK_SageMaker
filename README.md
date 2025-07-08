# Proyecto MLOps con AWS SageMaker y CDK

Este proyecto implementa un flujo completo de MLOps sobre AWS utilizando herramientas como SageMaker, AWS CLI, y AWS CDK, para un modelo de regresion. A continuación se detalla el procedimiento para la reproducción del pipeline, la estructura del proyecto y la configuración requerida.

---

## Requisitos Previos

### Instalación de AWS CLI

Para instalar AWS CLI:

```bash
pip install awscli
```

O visita: [https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)

### Instalación de AWS CDK

Para instalar AWS CDK:

```bash
npm install -g aws-cdk
```

O consulta: [https://docs.aws.amazon.com/cdk/v2/guide/home.html](https://docs.aws.amazon.com/cdk/v2/guide/home.html)

---

## Estructura del Proyecto

El proyecto fue creado con AWS CDK, definiendo inicialmente la infraestructura básica y progresivamente integrando scripts relevantes hasta llegar a la siguiente estructura:

### Scripts Relevantes:

* train.py: Entrena el modelo utilizando los datos procesados.
* processing.py: Limpia y transforma los datos raw para el entrenamiento.
* inference.py: Contiene la lógica para cargar el modelo y ejecutar predicciones en el endpoint.
* mps_group_stack.py: Script de infraestructura CDK que define buckets, roles y pipeline.

### Notebooks `.ipynb`

Fueron usados para pruebas locales y exploratorias del modelo y los datos.

---

## Conexión a AWS

Las rutas utilizadas en este proyecto están conectadas a la cuenta AWS del desarrollador original. Para adaptar este proyecto a una nueva cuenta:

1. Configura AWS CLI:

```bash
aws configure
```

2. Proporciona tus credenciales:

```
AWS Access Key ID
AWS Secret Access Key
Region (ej. us-east-1)
```

---

## Ampliación de Quotas

Es necesario **aumentar las quotas de procesamiento** en SageMaker desde la consola de AWS para asegurar que el pipeline funcione correctamente (ej. `ml.m5.large`).

---

## Despliegue de la Arquitectura

Luego de haber adaptado la cuenta:

```bash
cdk deploy
```

Esto creará los buckets y recursos necesarios como se define en `mps_group_stack.py`.

---

## Carga Manual de Datos y Scripts

1. Cargar datos raw:

   * Subir `data.csv` al bucket `RawDataBucket` manualmente.

2. Cargar script de procesamiento:

   * Subir `processing.py` al mismo bucket.

3. Salida de procesamiento:

   * El script genera la carpeta `processed/` dentro de `ProcessedDataBucket`, donde se almacenan los datos limpios.

4. Empaquetar scripts para entrenamiento e inferencia:

```bash
tar -czf code.tar.gz train.py inference.py
```

* Subir `code.tar.gz` manualmente a `ProcessedDataBucket`.

---

## Ejecución del Pipeline

Lanzar el pipeline desde consola o con el comando (si está configurado como trigger):

Verificar que el modelo fue creado:

```bash
aws s3 ls s3://<processed-bucket>/model/
```

---

## Activación Manual del Pipeline

Si deseas activar manualmente:

```bash
aws sagemaker start-pipeline-execution --pipeline-name mlops-mps-pipeline
```

---

## Creación del Modelo Manualmente

Ejemplo de creación del modelo desde artefacto en S3:

```bash
aws sagemaker create-model \
  --model-name mlops-mps-manual-model \
  --primary-container '{
    "Image": "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1",
    "ModelDataUrl": "s3://<processed-bucket>/model/mlops-mps-training-job-xyz/output/model.tar.gz",
    "Environment": {
      "SAGEMAKER_PROGRAM": "inference.py",
      "SAGEMAKER_SUBMIT_DIRECTORY": "s3://<processed-bucket>/model/mlops-mps-training-job-xyz/output/model.tar.gz"
    }
  }' \
  --execution-role-arn arn:aws:iam::<account-id>:role/<role-name> \
  --region us-east-1
```

---

## Configuración del Endpoint

```bash
aws sagemaker create-endpoint-config \
  --endpoint-config-name mlops-mps-endpoint-config-vX \
  --production-variants '[
    {
      "VariantName": "AllTraffic",
      "ModelName": "mlops-mps-manual-model",
      "InitialInstanceCount": 1,
      "InstanceType": "ml.m5.large"
    }
  ]' \
  --region us-east-1
```

---

## Creación del Endpoint

```bash
aws sagemaker create-endpoint \
  --endpoint-name mlops-mps-endpoint-vX \
  --endpoint-config-name mlops-mps-endpoint-config-vX \
  --region us-east-1
```

---

## Prueba del Endpoint

El archivo `test_prod_model.json` contiene los datos de entrada para probar el modelo:

```json
{
  "instances": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -0.2, 1.2, 0, 1, 0, 0, 0, 0]]
}
```

Invocar el endpoint:

```bash
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name mlops-mps-endpoint-vX \
  --body fileb://test_prod_model.json \
  --content-type application/json \
  --accept application/json \
  output.json
```

La salida de la predicción se guarda en el archivo `output.json`.

---

## Notas Finales

* Asegúrate de limpiar modelos, configuraciones y endpoints no utilizados para evitar cargos innecesarios.
* Personaliza las rutas de S3, nombres y roles según tu cuenta de AWS.

---


**YominJaramilloM**
