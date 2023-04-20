program main
  implicit none

  !> Initialization
  ! Input x
  real, dimension(3, 1) :: x
  ! Declare auxiliary matrices (weights and biases)
  real, dimension(600,3):: W0
  real, dimension(600,1) :: b0
  real, dimension(200,600) :: W2
  real, dimension(200,1):: b2
  real, dimension(1,200) :: W4
  real, dimension(1,1) :: b4
  ! Output y
  real :: p

  ! Declare other variables, such as filename and integers i, j for loops
  character(len=256) :: fname
  integer            :: i, j

  ! Declare the example values for D, S, tau we are going to test on:
  real :: D   = 10.204131145455385
  real :: S   = 12.026584842282125
  real :: tau = 22.131296926293793

  ! Prims to be computed:
  real :: rho
  real :: eps
  real :: v
  real :: W

  !> Interface for subroutine read matrix
  interface
    subroutine ReadMatrix(fname, nrows, ncols, matrix, un)
      implicit none

      ! Declare the variables
      character(len=256), intent(in)                  :: fname
      !character(len=256)                             :: str_line
      integer, intent(in)                             :: nrows, ncols
      integer, intent(in)                             :: un
      integer                                         :: iflag, nlines
      integer                                         :: i, j
      real, dimension(:), allocatable                 :: values
      real, dimension(:, :), allocatable, intent(out) :: matrix
      integer, dimension (1:2)                        :: order2 = (/ 2, 1 /)

    end subroutine
  end interface

  !> Read our matrices
  fname = "weight0_flat.csv"
  call ReadMatrix(fname, 600, 3, W0, 1)

  fname = "bias0_flat.csv"
  call ReadMatrix(fname, 600, 1, b0, 2)

  fname = "weight2_flat.csv"
  call ReadMatrix(fname, 200, 600, W2, 3)

  fname = "bias2_flat.csv"
  call ReadMatrix(fname, 200, 1, b2, 4)

  fname = "weight4_flat.csv"
  call ReadMatrix(fname, 1, 200, W4, 5)

  fname = "bias4_flat.csv"
  call ReadMatrix(fname, 1, 1, b4, 6)

  print *, "Reading: all done!"

  ! Make a prediction:
  x = reshape([D, S, tau], [3,1])
  print *, "We are predicting for:"
  print *, x

  call predict(x, p, W0, b0, W2, b2, W4, b4)

  print *, "Prediction"
  print *, p

  call computePrims(D, S, tau, p, rho, eps, v, W)

  print *, "rho"
  print *, rho
  print *, "eps"
  print *, eps

end program main

subroutine ReadMatrix(fname, nrows, ncols, matrix, un)
  implicit none

  ! Declare the variables
  character(len=256), intent(in)             :: fname
  character(len=256)                         :: str_line
  integer, intent(in)                        :: nrows, ncols
  integer, intent(in)                        :: un
  integer                                    :: iflag, nlines
  integer                                    :: i, j

  ! Specify the values
  nlines = nrows*ncols

  real, dimension(nlines)                    :: values
  real, dimension(nrows, ncols), intent(out) :: matrix

  integer, dimension (1:2) :: order2 = (/ 2, 1 /)

  ! Open the file for reading
  open(unit=un, file=fname, status='old')!, access='sequential')  !


  ! Allocate memory for values and matrix
  !allocate(values(nlines))
  !allocate(matrix(nrows, ncols))

  do i = 0,ncols
    do j = 0, nrows
      read(un,*,iostat=iflag) matrix(i, j)
      if (iflag/=0) exit
      !print *, i
      !print *, values(i)
    enddo
  enddo
  ! Reading is over, close the file
  close(unit = un)
  !   matrix = reshape(values, (/ nrows, ncols /), order=order2)
end subroutine ReadMatrix

! This is a subroutine which takes input values x as an array and computes their sigmoid function values
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

! This is a subroutine that, given the conserved variables and the pressure, computes the other primitive variables
subroutine computePrims(D, S, tau, p, rho, eps, v, W)
  implicit none

  !> Initialization
  real, intent(in)  :: D
  real, intent(in)  :: S
  real, intent(in)  :: tau
  real, intent(in)  :: p
  real, intent(out) :: rho
  real, intent(out) :: eps
  real, intent(out) :: v
  real, intent(out) :: W

  !> Do the calculations

  v = S/(tau + D + p)
  W = 1/sqrt(1-v**2)

  rho = D/W
  eps = (tau + D*(1-W) + p*(1-W**2))/(D*W)
end subroutine computePrims

! Predict refers to the neural network. Given x and the weight matrices of the neural net, the subroutine predicts the pressure
subroutine predict(x, p, W0, b0, W2, b2, W4, b4)
  implicit none

  ! Declare input and output types:
  real, intent(in)  :: x(3, 1)
  real              :: xx(600, 1)  ! intermediate result 1
  real              :: yy(600, 1)  ! intermediate result 1, with sigmoid
  real              :: xxx(200, 1) ! intermediate result 2
  real              :: yyy(200, 1) ! intermediate result 2, with sigmoid
  real              :: y(1, 1)     ! pressure, but as array
  real, intent(out) :: p           ! pressure, but as array

  ! Declare auxiliary matrices (weights and biases), also given as input
  real, intent(in) :: W0(600, 3)
  real, intent(in) :: b0(600, 1)
  real, intent(in) :: W2(200, 600)
  real, intent(in) :: b2(200, 1)
  real, intent(in) :: W4(1, 200)
  real, intent(in) :: b4(1, 1)

  !> Interface for sigmoid function
  ! Note: we apply this to a vector, but has shape (n, 1), with n the size of the vector
  ! Vector sigmoid function
  interface
    subroutine sigmoid(x, sigmoid_values)
      implicit none

      integer, dimension(2) :: s
      integer               :: i

      real, dimension(:,:), intent(in) :: x
      real, dimension(:,:), intent(out) :: sigmoid_values

    end subroutine sigmoid
  end interface

  ! Do the calculation:
  xx  = matmul(W0, x)   + b0
  call sigmoid(xx, yy)
  xxx = matmul(W2, yy)  + b2
  call sigmoid(xxx, yyy)
  y   = matmul(W4, yyy) + b4

  ! Get the end result as a scalar, not an array
  p = y(1, 1)
end subroutine predict
