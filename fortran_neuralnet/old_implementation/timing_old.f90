program timing_old
  
  use neuralnet
  
  implicit none

  ! Define the network variables
  double precision :: D, S, tau
  double precision :: p

  integer :: counter, n

  ! Initialize
  call grhd_phys_neuralnet_init('./')
  
  ! TODO test

  write(*, *) nn_tab%bias4

  counter = 0
  n = 10

  D = 10.20413115d0
  S = 12.02658484d0
  tau = 22.13129693d0

  call  nn_predict(D, S, tau, p)

  write(*, *) p

  do while (counter < n)
    !!! Testing init
    call grhd_phys_neuralnet_init('./')

    !!! Testing inference
    !call random_number(D)
    !call random_number(S)
    !call random_number(tau)
    !call nn_predict(D, S, tau, p)

    !!! Verbose:
    !write(*, *) counter
    !write(*, *) D, S, tau
    !write(*, *) p
    
    ! Increment counter
    counter = counter + 1
  end do


end program timing_old
