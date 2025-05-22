# Use AWS Lambda Python base image
FROM public.ecr.aws/lambda/python:3.9

# Set working directory
WORKDIR ${LAMBDA_TASK_ROOT}

# Copy requirements first for better caching
COPY lambda_requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r lambda_requirements.txt

# Copy function code and model files
COPY food_predictor/ ${LAMBDA_TASK_ROOT}/food_predictor/
COPY models ${LAMBDA_TASK_ROOT}/models 

# Copy any data files needed (uncomment if needed)
# COPY DB38.xlsx ${LAMBDA_TASK_ROOT}/

# Set the CMD to your existing handler
CMD [ "food_predictor.api.lambda_handler.lambda_handler" ]