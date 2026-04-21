import { useEffect, useState } from "react";
import type { Dispatch, SetStateAction } from "react";

type ToastType = "loading" | "success" | "error";

type Toast = {
  id: number;
  type: ToastType;
  message: string;
};

type PromiseMessages<T> = {
  loading: string;
  success: string | ((value: T) => string);
  error: string | ((error: unknown) => string);
};

let nextId = 1;
let setToastsRef: Dispatch<SetStateAction<Toast[]>> | null = null;

function upsertToast(toast: Toast) {
  setToastsRef?.((toasts) => {
    const existing = toasts.findIndex((item) => item.id === toast.id);
    if (existing === -1) return [...toasts, toast];
    return toasts.map((item) => (item.id === toast.id ? toast : item));
  });
}

function removeToast(id: number) {
  setToastsRef?.((toasts) => toasts.filter((toast) => toast.id !== id));
}

function resolveMessage<T>(message: string | ((value: T) => string), value: T) {
  return typeof message === "function" ? message(value) : message;
}

export const toaster = {
  promise<T>(promise: Promise<T>, messages: PromiseMessages<T>): Promise<T> {
    const id = nextId++;
    upsertToast({ id, type: "loading", message: messages.loading });

    return promise
      .then((value) => {
        upsertToast({ id, type: "success", message: resolveMessage(messages.success, value) });
        window.setTimeout(() => removeToast(id), 2800);
        return value;
      })
      .catch((error) => {
        upsertToast({
          id,
          type: "error",
          message: resolveMessage(messages.error, error),
        });
        window.setTimeout(() => removeToast(id), 7000);
        throw error;
      });
  },
};

export function Toaster() {
  const [toasts, setToasts] = useState<Toast[]>([]);

  useEffect(() => {
    setToastsRef = setToasts;
    return () => {
      setToastsRef = null;
    };
  }, []);

  return (
    <div className="toast-region" role="status" aria-live="polite">
      {toasts.map((toast) => (
        <div className={`toast toast-${toast.type}`} key={toast.id}>
          <p>{toast.message}</p>
        </div>
      ))}
    </div>
  );
}
