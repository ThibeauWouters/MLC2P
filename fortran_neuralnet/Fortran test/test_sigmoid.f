program main
  real, dimension(5) :: yprime
  real, dimension(5) :: y

  ! Vector sigmoid function
  interface
    subroutine sigmoid(x, sigmoid_values)
      implicit none

      integer :: s, i
      real, dimension(:), intent(in)  :: x
      real, dimension(:), intent(out) :: sigmoid_values

    end subroutine sigmoid
  end interface


end program main


subroutine sigmoid(x, sigmoid_values)
  implicit none

  integer :: s, i
  real, dimension(:), intent(in) :: x
  real, dimension(:), intent(out) :: sigmoid_values

  s = size(x)

  ! Fill array with sigmoid values
  do i = 1, s
      sigmoid_values(i) = 1 / (1 + exp(-x(i)))
  end do

end subroutine sigmoid
