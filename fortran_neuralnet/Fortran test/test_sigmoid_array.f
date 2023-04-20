program main
  !real, dimension(5) :: y
  real, dimension(5, 1) :: yarray, yprime

  ! Vector sigmoid function
  interface
    subroutine sigmoid(x, sigmoid_values)
      implicit none

      integer, dimension(2) :: s
      integer :: i
      real, dimension(:,:), intent(in) :: x
      real, dimension(:,:), intent(out) :: sigmoid_values

    end subroutine sigmoid
  end interface

  yarray = reshape([1,2,3,4,5], [5,1])

  print *, yarray

  call sigmoid(yarray, yprime)
  print *, yprime

end program main


subroutine sigmoid(x, sigmoid_values)
  implicit none

  integer, dimension(2) :: s
  integer :: i
  real, dimension(:,:), intent(in)  :: x
  real, dimension(:,:), intent(out) :: sigmoid_values

  s = shape(x)

  ! Fill array with sigmoid values
  do i = 1, s(1)
      sigmoid_values(i,1) = 1 / (1 + exp(-x(i,1)))
  end do

end subroutine sigmoid

!!! Sigmoid function for a single value:
!real function singlesigmoid(x)
!    implicit none
!    real, intent(in)  :: x
!    real, intent(out) :: singlesigmoid
!
!    singlesigmoid = 1 / (1 + exp(-x))
!end function singlesigmoid
