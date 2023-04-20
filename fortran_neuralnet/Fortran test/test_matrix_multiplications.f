program matmultest
  implicit none

  ! Declaration
  real :: matrix1(3, 3)
  real :: matrix2(3, 3)

  ! Initialization
  ! note: you can induce a line break providede you use the "&" symbol:
  matrix1 = reshape([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &
              [3,3])
  matrix2 = transpose(matrix1)

  ! Testing:
  print *, matrix1
  print *, matmul(matrix1, matrix2)

end program
