#include "header.h"

int main()
{
    EMBEDDING* model;
    init(model);
    int m = 10, n = 1;
    float **a = createArray(m, n, rand());
    display(a, m, n);
    return 0;
}


