!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine ryser(matrix, n, perm)

implicit none

integer,  parameter   :: dp = kind(1.0d0)

integer,  intent(in)  :: n
real(dp), intent(in)  :: matrix(n,n)

real(dp), intent(out) :: perm

logical               :: flag, gval, w_gray(n)
integer               :: gcur, sign, i, j
real(dp)              :: w_perm, w_sums(n)

gcur   = 0
gval   = .false.
perm   = 1.0d0
sign   = -1
w_gray = .false.
w_sums = 0.0d0

! compute the first unpermuted row sums
do j=1,n
    do i=1,n
        w_sums(i) = w_sums(i) + matrix(i,j)
    end do
end do

! compute the first product of row sum permutations
do i=1,n
    perm = perm * w_sums(i)
end do

! loop through all 2**n permutations of row sums
permutations: do i=2,2**n

    ! handle the gray code
    flag = .true.
    graycode: do j=1,n
        change_bit: if (flag) then

            ! determine which bit in the gray code word will change
            if (.not. w_gray(j)) then
                w_gray(j) = .true.
                flag      = .false.
                gcur      = j
            else
                w_gray(j) = .false.
            end if

            ! update the bit in the gray code
            if (gcur == n) then
                gval = w_gray(n)
            else
                gval = .not. (w_gray(gcur) .and. w_gray(gcur+1))
            end if

        end if change_bit
    end do graycode

    ! update the row sums according to the changed bit
    do j=1,n
        if (gval) then
            w_sums(j) = w_sums(j) - matrix(j,gcur)
        else
            w_sums(j) = w_sums(j) + matrix(j,gcur)
        end if
    end do

    ! compute the next product of row sum permutations
    w_perm = 1
    do j=1,n
        w_perm = w_perm * w_sums(j)
    end do
    perm = perm + sign * w_perm

    ! flip the sign
    sign = -sign

end do permutations

end subroutine ryser

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
