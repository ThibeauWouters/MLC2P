program main

    USE rosenna
    implicit none
    DOUBLE PRECISION, DIMENSION(1,3) :: inputs
    DOUBLE PRECISION, DIMENSION(1,1) :: output

    inputs = RESHAPE(    (/10.20413115, 12.02658484, 22.13129693/),    (/1, 3/), order =     [2 , 1 ])

    write(*, *) "Testing my own network:"

    CALL initialize()

    CALL use_model(inputs, output)

    open(1, file = "test.txt")
    WRITE(1, *) SHAPE(output)
    WRITE(1, *) PACK(RESHAPE(output,(/SIZE(output, dim = 2), SIZE(output, dim = 1)/), order = [2, 1]),.true.)
    print *, output

end program main
