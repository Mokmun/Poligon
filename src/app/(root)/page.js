import React from 'react'
import Link from 'next/link'
import Image from 'next/image'

const main =  () => {
  return (
    <div className="flex flex-row w-full items-left px-36 justify-center h-dvh">
        <div className="flex flex-col items-start justify-center h-full w-3/5">
          <div className="flex flex-col items-start gap-6 justify-start w-full">
            <h1 className="text-5xl font-bold">Build your 3D model</h1>
            <p className="text-white text-base/7 tracking-wide">
              Create and view 3D models
              and explore a wide range of furniture designs. Our platform allows you to customize and visualize your own creations in a 3D environment, 
              making it easier to bring your ideas to life.
            </p>
          </div>
          <div className='flex flex-row gap-4 mt-10'>
            <button className="btn-primary text-white text-lg px-9 py-3 rounded-xl">
              <Link href='/create3D'>
                Start Generating
              </Link>
            </button>
          </div>
        </div>
          <Image src='/assets/star.png' alt="hero" width={500} height={500} className='object-contain' />
    </div>
  )
}

export default main