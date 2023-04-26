program net

    implicit none 

    double precision :: W0(50, 3)
    double precision :: W2(50, 50)
    double precision :: W4(3, 50)

    double precision :: b0(50)
    double precision :: b2(50)

    double precision :: x(3)  ! input
    double precision :: xx(50)  ! intermediate result 1, after first hidden layer
    double precision :: yy(50)  ! intermediate result 1, applied sigmoid
    double precision :: xxx(50) ! intermediate result 2, after second hidden layer
    double precision :: yyy(50) ! intermediate result 2, after sigmoid
    double precision :: y(3)    ! output as array, three output nodes

    !> Timing:
    real :: start, finish
    integer :: i, counter
    integer :: n_samples = 1000000

    call random_number(W0)
    call random_number(W2)
    call random_number(W4)
    
    call random_number(b0)
    call random_number(b2)

    counter = 0

    call cpu_time(start)
    do i = 1, n_samples
        call random_number(x)
        y = 0.0d0
        call random_number(b0)
        call random_number(b2)

        ! Do the calculation:
        CALL DGEMV('N', 50, 3, 1.0d0, W0, 50, x, 1, 1.0d0, b0, 1)
        xx = b0
        call relu(xx, yy)
        CALL DGEMV('N', 50, 50, 1.0d0, W2, 50, yy, 1, 1.0d0, b2, 1)
        xxx = b2
        call relu(xxx, yyy)
        CALL DGEMV('N', 3, 50, 1.0d0, W4, 3, yyy, 1, 0.0d0, y, 1)

        counter = counter + 1
    end do
    call cpu_time(finish)
    print '("Time = ",f6.3," seconds.")',finish-start
    
    write(*, *) counter
    write(*, *) y

contains 

    subroutine relu(x, relu_values)
    implicit none
    
    double precision, dimension(:), intent(in)  :: x
    double precision, intent(out)               :: relu_values(size(x))

    relu_values = max(0.0d0, x)

  end subroutine relu

end program net