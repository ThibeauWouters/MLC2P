PROGRAM dgemv_test

    IMPLICIT NONE
    
    INTEGER, PARAMETER :: n = 50, m = 3
    DOUBLE PRECISION, DIMENSION(n,m) :: A
    DOUBLE PRECISION, DIMENSION(m) :: x, B
    INTEGER :: i

    ! Initialize A and x with random values
    CALL RANDOM_NUMBER(A)
    CALL RANDOM_NUMBER(x)
    
    ! Initialize B to all ones
    !CALL RANDOM_NUMBER(B)
    B = 1.0d0
    
    ! Call dgemv to compute A*x + B
    CALL DGEMV('N', n, m, 1.0d0, A, n, x, 1, 1.0d0, B, 1)
    
    ! Print the resulting vector to the screen
     WRITE(*,*) "Resulting vector:"
     DO i = 1, n
         WRITE(*,*) B(i)
     END DO

END PROGRAM dgemv_test
