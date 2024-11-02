(define (problem game-1)
  (:domain game)
  (:objects
    l0 - coord
  )
  (:init
    (on l0 l0 Y)
    (max l0)
    (min l0)
  )
  (:htn
    :tasks (and
      (play)
    )
  )
  (:goal (and
    (on l0 l0 N)
  ))
)
