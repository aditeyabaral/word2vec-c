#include "header.h"

float cost(EMBEDDING* model)
{
    float loss = 0;
    int m = model->batch_size;
    float sum;
    for (int i=0; i<m; i++)
    {
        sum = 0;
        for(int j = 0; j<model->vocab_size; j++)
            sum+= (model->Y[j][i])*log(model->yhat[j][i]) + (1-model->Y[j][i])*log(1-model->yhat[j][j]);
        loss+= sum;
    }
    loss = (-1.0/m)*loss;
    return loss;
}

void forward_propagation(EMBEDDING* model)
{
    double **W1X = multiply(model->W1, model->X, model->dimension, model->vocab_size, model->vocab_size, model->batch_size);
    model->Z1 = broadcast_and_add(W1X, model->b1, model->dimension, model->batch_size, model->dimension, 1);
    model->A1 = relu(model->Z1, model->dimension, model->batch_size);
    
    double **W2A1 = multiply(model->W2, model->A1, model->vocab_size, model->dimension, model->dimension, model->batch_size);
    model->Z2 = broadcast_and_add(W2A1, model->b2, model->vocab_size, model->batch_size, model->vocab_size, 1);
    model->yhat = softmax(model->Z2, model->vocab_size, model->batch_size, 1);
    free2D(W1X, model->dimension, model->batch_size);
    free2D(W2A1, model->vocab_size, model->batch_size);
    #if 0
    printf("W1X:\n");
    displayArray(W1X, model->dimension, model->batch_size);
    printf("Z1:\n");
    displayArray(model->Z1, model->dimension, model->batch_size);
    printf("A1:\n");
    displayArray(model->A1, model->dimension, model->batch_size);
    printf("Z2:\n");
    displayArray(model->Z2, model->vocab_size, model->batch_size);
    printf("A2: \n");
    displayArray(model->yhat, model->vocab_size, model->batch_size);
    #endif
}

void back_propagation(EMBEDDING* model)
{
    double **W2T = transpose(model->W2, model->vocab_size, model->dimension);
    double **yhat_diff_y = subtract(model->yhat, model->Y, model->vocab_size, model->batch_size);
    double **W2T_mul_y_hat_diff_y = multiply(W2T, yhat_diff_y, model->dimension, model->vocab_size, model->vocab_size, model->batch_size);
    double **OnesVector = createOnesArray(model->batch_size, 1);
    double ratio = 1.0/model->batch_size;

    double** dW1 = relu(W2T_mul_y_hat_diff_y, model->dimension, model->batch_size);
    dW1 = multiply(dW1, transpose(model->X, model->vocab_size, model->batch_size), model->dimension, model->batch_size, model->batch_size, model->vocab_size);
    dW1 = multiply_scalar(dW1, ratio, model->dimension, model->vocab_size);

    double** AT = transpose(model->A1, model->dimension, model->batch_size);
    double** dW2 = multiply(yhat_diff_y, AT, model->vocab_size, model->batch_size, model->batch_size, model->dimension);
    dW2 = multiply_scalar(dW2, ratio, model->vocab_size, model->dimension);

    double** db1 = relu(W2T_mul_y_hat_diff_y, model->dimension, model->batch_size);
    db1 = multiply(db1, OnesVector, model->dimension, model->batch_size, model->batch_size, 1);
    db1 = multiply_scalar(db1, ratio, model->dimension, 1);

    double** db2 = multiply(yhat_diff_y, OnesVector, model->vocab_size, model->batch_size, model->batch_size, 1);
    db2 = multiply_scalar(db2, ratio, model->vocab_size, 1);

    double** alpha_dW1 = multiply_scalar(dW1, model->alpha, model->dimension, model->vocab_size);
    model->W1 = subtract(model->W1, alpha_dW1, model->dimension, model->vocab_size);

    double** alpha_dW2 = multiply_scalar(dW2, model->alpha, model->vocab_size, model->dimension);
    model->W2 = subtract(model->W2, alpha_dW2, model->vocab_size, model->dimension);

    double** alpha_db1 = multiply_scalar(db1, model->alpha, model->dimension, 1);
    model->b1 = subtract(model->b1, alpha_db1, model->dimension, 1);

    double** alpha_db2 = multiply_scalar(db2, model->alpha, model->vocab_size, 1);
    model->b2 = subtract(model->b2, alpha_db2, model->vocab_size, 1);

    #if 0
    printf("dW1: \n");
    displayArray(dW1, model->dimension, model->vocab_size);
    printf("dW2: \n");
    displayArray(dW2, model->vocab_size, model->dimension);
    printf("db1: \n");
    displayArray(db1, model->dimension, 1);
    printf("db2 \n");
    displayArray(db2, model->vocab_size, 1);
    printf("After subtraction:\n\n");
    printf("\nW1: \n");
    displayArray(model->W1, model->dimension, model->vocab_size);
    printf("\nW2: \n");
    displayArray(model->W2, model->vocab_size, model->dimension);
    printf("\nb1: \n");
    displayArray(model->b1, model->dimension, 1);
    printf("\nb2: \n");
    displayArray(model->b2, model->vocab_size, 1);
    #endif

    free2D(W2T, model->dimension, model->vocab_size);
    free2D(yhat_diff_y, model->vocab_size, model->batch_size);
    free2D(W2T_mul_y_hat_diff_y, model->dimension, model->batch_size);
    free2D(OnesVector, model->batch_size, 1);
    free2D(dW1, model->dimension, model->vocab_size);
    free2D(AT, model->batch_size, model->dimension);
    free2D(dW2, model->vocab_size, model->dimension);
    free2D(db1, model->dimension, 1);
    free2D(db2, model->vocab_size, 1);
    free2D(alpha_dW1, model->dimension, model->vocab_size);
    free2D(alpha_dW2, model->vocab_size, model->dimension);
    free2D(alpha_db1, model->dimension, 1);
    free2D(alpha_db2, model->vocab_size, 1);
}

void gradientDescent(EMBEDDING* model)
{
    double loss;
    for(int i=0; i < model->epochs; i++)
    {
        forward_propagation(model);
        loss = cost(model);
        printf("Epoch: %d Loss: %lf\n", i, loss);
        back_propagation(model);
    }
    printf("Training completed.\n\n");
}