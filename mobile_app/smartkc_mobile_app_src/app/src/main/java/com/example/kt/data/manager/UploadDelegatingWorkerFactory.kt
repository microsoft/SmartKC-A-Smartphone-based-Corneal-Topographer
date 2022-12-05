package com.example.kt.data.manager

import androidx.work.DelegatingWorkerFactory
import com.example.kt.data.repo.FileRepository
import javax.inject.Inject

class UploadDelegatingWorkerFactory
@Inject
constructor(
    fileRepository: FileRepository
) : DelegatingWorkerFactory() {
    init {
        addFactory(WorkerFactory(fileRepository))
        // Add here other factories that you may need in your application
    }
}