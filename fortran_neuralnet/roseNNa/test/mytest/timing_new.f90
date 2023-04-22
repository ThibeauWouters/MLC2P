program timing_old

  use rosenna

  implicit none

  ! Define the network variables
  double precision :: D, S, tau
  double precision :: p

  double precision, DIMENSION(1,3) :: inputs
  double precision, DIMENSION(1,1) :: output

  integer :: counter, n

  ! Initialize
  call rosenna_initialize()

  counter = 0
  n = 10

  !D = 10.20413115d0
  !S = 12.02658484d0
  !tau = 22.13129693d0

  !inputs = RESHAPE((/ D, S, tau /), (/1,3/), order= [2,1])

  !call use_model(inputs, output)

  !p = output(1,1)

  !write(*, *) p


  do while (counter < n)
    
    !!! Timing init

    call rosenna_initialize()


    !!! Timing inference
    !call random_number(D)
    !call random_number(S)
    !call random_number(tau)

    !inputs = RESHAPE((/ D, S, tau/),  (/1, 3/), order =     [2 , 1 ])

    !call use_model(inputs, output)

    !!! Convert output to scalar for printing
    !p = output(1,1)

    !write(*, *) counter
    !write(*, *) D, S, tau
    !write(*, *) p

    ! Increment counter
    counter = counter + 1
  end do


end program timing_old
