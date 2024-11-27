import React, { useState, useEffect, useRef } from 'react';

interface AnimatedStatProps {
  end: string | number;
  duration?: number;
  label: string;
}

const AnimatedStat: React.FC<AnimatedStatProps> = ({ end, duration = 2000, label }) => {
  const [count, setCount] = useState("0");
  const elementRef = useRef<HTMLDivElement>(null);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
        }
      },
      { threshold: 0.1 }
    );

    if (elementRef.current) {
      observer.observe(elementRef.current);
    }

    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!isVisible) return;

    // Special case for "24/7"
    if (end === "24/7") {
      let startTimestamp: number;
      
      const step = (timestamp: number) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        
        if (progress < 1) {
          // Generate random numbers for both parts
          const hours = Math.floor(Math.random() * 24);
          const days = Math.floor(Math.random() * 7);
          setCount(`${hours}/${days}`);
          requestAnimationFrame(step);
        } else {
          setCount("24/7");
        }
      };

      requestAnimationFrame(step);
      return;
    }

    // Handle other cases (percentages and numbers)
    let startTimestamp: number;
    const endNum = typeof end === 'number' ? end : parseInt(end.replace(/\D/g, ''));
    const suffix = typeof end === 'string' ? end.replace(/[0-9]/g, '') : '';

    const step = (timestamp: number) => {
      if (!startTimestamp) startTimestamp = timestamp;
      const progress = Math.min((timestamp - startTimestamp) / duration, 1);
      
      const currentCount = Math.floor(progress * endNum);
      setCount(`${currentCount}${suffix}`);

      if (progress < 1) {
        requestAnimationFrame(step);
      }
    };

    requestAnimationFrame(step);
  }, [end, duration, isVisible]);

  return (
    <div ref={elementRef} className="text-center">
      <div className="text-4xl font-bold text-primary mb-2">{count}</div>
      <div className="text-gray-400">{label}</div>
    </div>
  );
};

export default AnimatedStat; 