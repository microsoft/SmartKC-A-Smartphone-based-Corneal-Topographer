package com.example.kt

import android.app.Application
import androidx.work.Configuration
import com.example.kt.data.manager.UploadDelegatingWorkerFactory
import dagger.hilt.android.HiltAndroidApp
import javax.inject.Inject

@HiltAndroidApp
class KTApp: Application(), Configuration.Provider {

    @Inject
    lateinit var workerFactory: UploadDelegatingWorkerFactory

    override fun getWorkManagerConfiguration(): Configuration =
        Configuration.Builder().setMinimumLoggingLevel(android.util.Log.DEBUG).setWorkerFactory(workerFactory).build()
}