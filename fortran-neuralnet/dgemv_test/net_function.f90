program net_function

    implicit none 

    double precision :: W0(50, 3)
    double precision :: W2(50, 50)
    double precision :: W4(3, 50)

    double precision :: b0(50, 1)
    double precision :: b2(50, 1)

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

    counter = 0

    call random_number(W0)
    call random_number(W2)
    call random_number(W4)
    
    call random_number(b0)
    call random_number(b2)

    call cpu_time(start)
    do i = 1, n_samples
      call random_number(x)

      ! Do the calculation:
      xx  = matmul(W0, x)  + b0(:, 1)
      yy = relu(xx)
      xxx = matmul(W2, yy) + b2(:, 1)
      yyy = relu(xxx)
      y   = matmul(W4, yyy)
      counter = counter + 1

    end do
    call cpu_time(finish)
    print '("Time = ",f6.3," seconds.")',finish-start
    
    write(*, *) counter
    write(*, *) y

contains 

    function relu(xvals) result(relu_values)
    implicit none
    
    double precision, dimension(:), intent(in)  :: xvals
    double precision                            :: relu_values(size(xvals))

    relu_values = max(0.0d0, xvals)

    end function relu

end program net_function